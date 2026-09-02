function void fl_solver_euler_halo_state_pack_send_data(FL_Solver_Euler *euler, FL_State *state) {
  profiler_begin_function();
  UG_Mesh *mesh = euler->mesh;
  for Iter_Range(it_gather, lane_range(mesh->sends.len)) {
    U32 cell_gather = mesh->sends.cell_send[it_gather];
    for Iter_Index(it_state, 5) {
      euler->halo_state_send_dat[it_gather * 5 + it_state] = state->states[it_state][cell_gather];
    }
  }
  lane_barrier();
  profiler_end_function();
}

function void fl_solver_euler_halo_state_unpack_receive_data(FL_Solver_Euler *euler, FL_State *state) {
  profiler_begin_function();
  UG_Mesh *mesh = euler->mesh;
  for Iter_Range(it_halo, lane_range(mesh->halos.len)) {
    for Iter_Index(it_state, 5) {
      state->states[it_state][mesh->cells.len + it_halo] = euler->halo_state_receive_dat[it_halo * 5 + it_state];
    }
  }
  lane_barrier();
  profiler_end_function();
}

function void fl_solver_euler_halo_gradient_limiter_pack_send_data(FL_Solver_Euler *euler, FL_Gradient_State *grad, FL_Limiter_State *lim) {
  profiler_begin_function();
  UG_Mesh *mesh = euler->mesh;
  for Iter_Range(it_gather, lane_range(mesh->sends.len)) {
    U32 cell_gather = mesh->sends.cell_send[it_gather];
    U64 base = it_gather * 20;
    for Iter_Index(it_state, 5) {
      for Iter_Index(it_component, 3) {
        euler->halo_gradient_limiter_send_dat[base + it_state * 3 + it_component] =
            grad->states[it_state].grad_dat[it_component][cell_gather];
      }
    }
    for Iter_Index(it_state, 5) {
      euler->halo_gradient_limiter_send_dat[base + 15 + it_state] = lim->states[it_state][cell_gather];
    }
  }
  lane_barrier();
  profiler_end_function();
}

function void fl_solver_euler_halo_gradient_limiter_unpack_receive_data(FL_Solver_Euler *euler, FL_Gradient_State *grad, FL_Limiter_State *lim) {
  profiler_begin_function();
  UG_Mesh *mesh = euler->mesh;
  for Iter_Range(it_halo, lane_range(mesh->halos.len)) {
    U64 base = it_halo * 20;
    for Iter_Index(it_state, 5) {
      for Iter_Index(it_component, 3) {
        grad->states[it_state].grad_dat[it_component][mesh->cells.len + it_halo] =
            euler->halo_gradient_limiter_receive_dat[base + it_state * 3 + it_component];
      }
    }
    for Iter_Index(it_state, 5) {
      lim->states[it_state][mesh->cells.len + it_halo] = euler->halo_gradient_limiter_receive_dat[base + 15 + it_state];
    }
  }
  lane_barrier();
  profiler_end_function();
}

function void fl_solver_euler_halo_state_build_request_list(FL_Solver_Euler *euler) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  ipc_rank_request_list_init(&euler->halo_state_request_list);

  for Iter_Index(it_rank, mesh->halos.block_len) {
    if (it_rank != ipc_rank_index()) {
      Range1_U64  halo_range  = mesh->halos.block_range[it_rank];
      U64         halo_len    = range1_u64_len(halo_range);

      if (halo_len) {
        // NOTE(cmat): Receive halo data from neighbours.
        U64 own_tag = ipc_rank_index();
        ipc_rank_record_receive(&euler->halo_state_request_list, halo_len * 5 * sizeof(F32), euler->halo_state_receive_dat + 5 * halo_range.min, it_rank, own_tag);
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
        ipc_rank_record_send(&euler->halo_state_request_list, send_len * 5 * sizeof(F32), euler->halo_state_send_dat + 5 * send_range.min, it_rank, other_tag);
      }
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_euler_halo_gradient_limiter_build_request_list(FL_Solver_Euler *euler) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  ipc_rank_request_list_init(&euler->halo_gradient_limiter_request_list);

  for Iter_Index(it_rank, mesh->halos.block_len) {
    if (it_rank != ipc_rank_index()) {
      Range1_U64  halo_range  = mesh->halos.block_range[it_rank];
      U64         halo_len    = range1_u64_len(halo_range);

      if (halo_len) {
        // NOTE(cmat): Receive halo data from neighbours.
        U64 own_tag = ipc_rank_index();
        ipc_rank_record_receive(&euler->halo_gradient_limiter_request_list, halo_len * 5 * sizeof(F32) * (3 + 1), euler->halo_gradient_limiter_receive_dat + 5 * halo_range.min * (3 + 1), it_rank, own_tag);
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
        ipc_rank_record_send(&euler->halo_gradient_limiter_request_list, send_len * 5 * sizeof(F32) * (3 + 1), euler->halo_gradient_limiter_send_dat + 5 * send_range.min * (3 + 1), it_rank, other_tag);
      }
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_euler_init(FL_Solver_Euler *euler, FL_Boundary_Map *boundary, FL_Scale scale, FL_Material material, V3F gravity, UG_Mesh *mesh, Arena *arena) {
  Zero_Fill(euler);

  euler->mesh                 = mesh;
  euler->boundary             = boundary;
  euler->scale                = scale;

  fl_state_init(&euler->flow_1,   material, mesh, 1, arena);
  fl_state_init(&euler->flow_2,   material, mesh, 1, arena);
  fl_state_init(&euler->residual, material, mesh, 0, arena);

  fl_gradient_state_init  (&euler->gradient, mesh, 1, arena);
  fl_limiter_state_init   (&euler->limiter,  mesh, 1, arena);

  euler->halo_state_receive_len     = 5 * mesh->halos.len;
  euler->halo_state_send_len        = 5 * mesh->sends.len;

  euler->halo_gradient_limiter_receive_len  = 5 * mesh->halos.len * (3 + 1);
  euler->halo_gradient_limiter_send_len     = 5 * mesh->sends.len * (3 + 1);

  if (lane_index() == 0) {
    euler->primitive_pressure          = arena_push_count(arena, F32, euler->flow_1.inner_len + euler->flow_1.halo_len + euler->flow_1.ghost_len);
    euler->primitive_v_x               = arena_push_count(arena, F32, euler->flow_1.inner_len + euler->flow_1.halo_len + euler->flow_1.ghost_len);
    euler->primitive_v_y               = arena_push_count(arena, F32, euler->flow_1.inner_len + euler->flow_1.halo_len + euler->flow_1.ghost_len);
    euler->primitive_v_z               = arena_push_count(arena, F32, euler->flow_1.inner_len + euler->flow_1.halo_len + euler->flow_1.ghost_len);

    euler->cell_time_step              = arena_push_count(arena, F64, mesh->cells.len);
    euler->lane_time_step              = arena_push_count(arena, F64, lane_count());
    euler->lane_state_norm2            = arena_push_count(arena, V3_F64, lane_count());

    euler->halo_state_receive_dat      = arena_push_count(arena, F32, euler->halo_state_receive_len);
    euler->halo_state_send_dat         = arena_push_count(arena, F32, euler->halo_state_send_len);

    euler->halo_gradient_limiter_receive_dat   = arena_push_count(arena, F32, euler->halo_gradient_limiter_receive_len);
    euler->halo_gradient_limiter_send_dat      = arena_push_count(arena, F32, euler->halo_gradient_limiter_send_len);
  }

  lane_broadcast_ptr(&euler->primitive_pressure, 0);
  lane_broadcast_ptr(&euler->primitive_v_x,      0);
  lane_broadcast_ptr(&euler->primitive_v_y,      0);
  lane_broadcast_ptr(&euler->primitive_v_z,      0);

  lane_broadcast_ptr(&euler->cell_time_step,              0);
  lane_broadcast_ptr(&euler->lane_time_step,              0);
  lane_broadcast_ptr(&euler->lane_state_norm2,            0);
  lane_broadcast_ptr(&euler->halo_state_send_dat,         0);
  lane_broadcast_ptr(&euler->halo_state_receive_dat,      0);
  lane_broadcast_ptr(&euler->halo_gradient_limiter_send_dat,      0);
  lane_broadcast_ptr(&euler->halo_gradient_limiter_receive_dat,   0);

  // NOTE(cmat): Build request list for halo synchronization.
  fl_solver_euler_halo_state_build_request_list             (euler);
  fl_solver_euler_halo_gradient_limiter_build_request_list  (euler);

  // TODO(cmat): Temporary.
  euler->gravity = gravity;
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

    V3F inner_center      = mesh->cells.center[cell_parent_index];
    V3F ghost_center      = mesh->cells.center[flow_ghost_index];

    UG_Cell_Faces *faces  = &mesh->cells.faces[cell_parent_index];
    V3F normal            = v3f(faces->normal_x[face_parent_index], faces->normal_y[face_parent_index], faces->normal_z[face_parent_index]);
    V5F inner_state       = fl_state_get(state, cell_parent_index);
    V5F ghost_state       = fl_boundary_map_ghost(euler->boundary, marker_index, inner_state, inner_center, ghost_center, normal, &euler->scale, &state->material, euler->gravity);

    fl_state_set(state, flow_ghost_index, ghost_state);
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_euler_compute_primitive_range(FL_Solver_Euler *euler, FL_State *state, Range1_U64 range) {
  profiler_begin_function();

  U64 range_len = range1_u64_len(range);
  UG_Mesh *mesh = euler->mesh;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64 it_cell                         = range.min + it_range;
    F32 rho_rcp                         = 1.f / state->rho[it_cell];
    F32 vx                              = state->rho_v1[it_cell] * rho_rcp;
    F32 vy                              = state->rho_v2[it_cell] * rho_rcp;
    F32 vz                              = state->rho_v3[it_cell] * rho_rcp;
    F32 kinetic_energy                  = 0.5f * (state->rho_v1[it_cell] * vx + state->rho_v2[it_cell] * vy + state->rho_v3[it_cell] * vz);
    F32 pressure                        = (state->material.gamma - 1.f) * (state->energy[it_cell] - kinetic_energy);
    euler->primitive_pressure [it_cell] = pressure;
    euler->primitive_v_x      [it_cell] = vx;
    euler->primitive_v_y      [it_cell] = vy;
    euler->primitive_v_z      [it_cell] = vz;
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_compute_residual_range(FL_Solver_Euler *euler, FL_State *state, FL_State *residual, FL_Gradient_State *grad, Range1_U64 range, B32 compute_time_step, F64 *cell_time_step) {
  profiler_begin_function();

  U64 range_len = range1_u64_len(range);
  UG_Mesh *mesh = euler->mesh;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64            it_cell        = range.min + it_range;
    UG_Cell_Faces *faces          = &mesh->cells.faces[it_cell];
    V5F            cell_residual  = v5f(0, 0, 0, 0, 0);

    V5F left_primitive = {
      .x1 = state->rho[it_cell],
      .x2 = euler->primitive_v_x[it_cell],
      .x3 = euler->primitive_v_y[it_cell],
      .x4 = euler->primitive_v_z[it_cell],
      .x5 = euler->primitive_pressure[it_cell],
    };

    // NOTE(cmat): Primitive gradients. We're already storing these as primitive values.
    V3F left_grad[5] = { };
    for Iter_Index(it_state, 5) {
      left_grad[it_state] = v3f(grad->states[it_state].grad_x[it_cell], grad->states[it_state].grad_y[it_cell], grad->states[it_state].grad_z[it_cell]);
    }

    V3F cell_center   = mesh->cells.center[it_cell];
    F32 cell_volume   = mesh->cells.volume[it_cell];

    F64 spectral_inviscid_sum = 0.f;
    F64 spectral_viscous_sum  = 0.f;
    for Iter_Index(it_face, 4) {
      U32 adjacent     = faces->adjacent[it_face];
      V3F normal       = v3f(faces->normal_x[it_face], faces->normal_y[it_face], faces->normal_z[it_face]);
      F32 area         = faces->area[it_face];
      V3F face_center  = v3f(faces->center_x[it_face], faces->center_y[it_face], faces->center_z[it_face]);

      // NOTE(cmat): Reconstruct left primitive state.
      F32 left_face_primitive[5] = { };
      V3F center_delta = v3f_sub(face_center, mesh->cells.center[it_cell]);
      for Iter_Index(it_state, 5) {
        F32 cell_limiter  = euler->limiter.states[it_state][it_cell];
        left_face_primitive[it_state] = left_primitive.dat[it_state] + cell_limiter * v3f_dot(left_grad[it_state], center_delta);
      }

      // NOTE(cmat): Convert left primitive state to conservative state.
      V3F left_momentum = v3f_mul(left_face_primitive[0], v3f(left_face_primitive[1], left_face_primitive[2], left_face_primitive[3]));
      V5F left_state    = v5f(left_face_primitive[0], left_momentum.x, left_momentum.y, left_momentum.z, fl_state_energy_from_pressure(&state->material, left_face_primitive[0], left_momentum, left_face_primitive[4]));

      V5F right_primitive = {
        .x1 = state->rho[adjacent],
        .x2 = euler->primitive_v_x[adjacent],
        .x3 = euler->primitive_v_y[adjacent],
        .x4 = euler->primitive_v_z[adjacent],
        .x5 = euler->primitive_pressure[adjacent],
      };

      // NOTE(cmat): Construct right state.
      // NOTE(cmat): If the right cell is not a ghost state, it has a gradient.
      V5F right_state   = { };
      V3F right_grad[5] = { };

      if (adjacent < (mesh->cells.len + mesh->halos.len)) {
        for Iter_Index(it_state, 5) {
          right_grad[it_state] = v3f(grad->states[it_state].grad_x[adjacent], grad->states[it_state].grad_y[adjacent], grad->states[it_state].grad_z[adjacent]);
        }

        // NOTE(cmat): Construct right primitive state with gradient.
        F32 right_face_primitive[5] = { };
        V3F center_delta_r = v3f_sub(face_center, mesh->cells.center[adjacent]);
        for Iter_Index(it_state, 5) {
          F32 limiter                    = euler->limiter.states[it_state][adjacent];
          right_face_primitive[it_state] = right_primitive.dat[it_state] + limiter * v3f_dot(right_grad[it_state], center_delta_r);
        }

        // NOTE(cmat): Convert right primitive state to conservative state.
        V3F right_momentum  = v3f_mul(right_face_primitive[0], v3f(right_face_primitive[1], right_face_primitive[2], right_face_primitive[3]));
        right_state         = v5f(right_face_primitive[0], right_momentum.x, right_momentum.y, right_momentum.z, fl_state_energy_from_pressure(&state->material, right_face_primitive[0], right_momentum, right_face_primitive[4]));

      // NOTE(cmat): If the right cell is a ghost state, it has a gradient zero. We fallback to first order.
      } else {
        // NOTE(cmat): Zero-out gradient.
        Stack_Array_Zero(right_grad);

        right_state = fl_state_get(state, adjacent);
      }

      F32 right_volume = mesh->cells.volume[adjacent];

      FL_Flux flux_inviscid   = fl_flux_hllc                    (left_state, right_state, normal, state->material.gamma);
      FL_Flux flux_viscous    = fl_flux_viscous_smagorinsky_LES (left_primitive, left_grad, cell_center, right_primitive, right_grad, mesh->cells.center[adjacent], normal, area, cell_volume, right_volume, &state->material);
      V5F     flux_total      = v5f_sub(flux_inviscid.state, flux_viscous.state);
      cell_residual           = v5f_sub(cell_residual, v5f_mul(area, flux_total));
      spectral_inviscid_sum  += (F64)area * (F64)flux_inviscid.lambda_max;
      spectral_viscous_sum   +=(F64)flux_viscous.lambda_viscous;
    }

    F32 volume     = mesh->cells.volume[it_cell];
    F32 volume_rcp = 1.f / volume;

    // NOTE(cmat): Gravity source term.
    // TODO(cmat): Temporary.
    V5F source_term = {
      .x1 = 0,
      .x2 = state->rho[it_cell] * euler->gravity.x,
      .x3 = state->rho[it_cell] * euler->gravity.y,
      .x4 = state->rho[it_cell] * euler->gravity.z,
      .x5 = state->rho[it_cell] * v3f_dot(left_primitive.x234, euler->gravity),
    };

    residual->rho     [it_cell] = cell_residual.x1 * volume_rcp + source_term.x1;
    residual->rho_v1  [it_cell] = cell_residual.x2 * volume_rcp + source_term.x2;
    residual->rho_v2  [it_cell] = cell_residual.x3 * volume_rcp + source_term.x3;
    residual->rho_v3  [it_cell] = cell_residual.x4 * volume_rcp + source_term.x4;
    residual->energy  [it_cell] = cell_residual.x5 * volume_rcp + source_term.x5;

    if (compute_time_step) {
      cell_time_step[it_cell] = (F64)volume / (spectral_inviscid_sum + spectral_viscous_sum);
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_compute_gradient_range(FL_Solver_Euler *euler, FL_State *state, FL_Gradient_State *gradient, Range1_U64 range) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);
  UG_Mesh *mesh = euler->mesh;

  for Iter_Range(it_range, lane_range(range_len)) {
    U64               it_cell   = range.min + it_range;
    UG_Cell_Faces    *faces     = &mesh->cells.faces[it_cell];
    UG_Cell_Gradient *grad_geo  = &mesh->cells.gradients[it_cell];

    V3F b[5] = { };

    V5F self_prim = {
      .x1 = state->rho[it_cell],
      .x2 = euler->primitive_v_x[it_cell],
      .x3 = euler->primitive_v_y[it_cell],
      .x4 = euler->primitive_v_z[it_cell],
      .x5 = euler->primitive_pressure[it_cell],
    };

    for Iter_Index(it_face, 4) {
      U32 adjacent = faces->adjacent[it_face];
      V5F adj_prim = {
        .x1 = state->rho[adjacent],
        .x2 = euler->primitive_v_x[adjacent],
        .x3 = euler->primitive_v_y[adjacent],
        .x4 = euler->primitive_v_z[adjacent],
        .x5 = euler->primitive_pressure[adjacent],
      };

      V3F w = grad_geo->weight_dx[it_face];
      for Iter_Index(it_state, 5) {
        b[it_state] = v3f_add(b[it_state], v3f_mul(adj_prim.dat[it_state] - self_prim.dat[it_state], w));
      }
    }

    for Iter_Index(it_state, 5) {
      gradient->states[it_state].grad_x[it_cell] = grad_geo->inv_A_xx * b[it_state].x + grad_geo->inv_A_xy * b[it_state].y + grad_geo->inv_A_xz * b[it_state].z;
      gradient->states[it_state].grad_y[it_cell] = grad_geo->inv_A_xy * b[it_state].x + grad_geo->inv_A_yy * b[it_state].y + grad_geo->inv_A_yz * b[it_state].z;
      gradient->states[it_state].grad_z[it_cell] = grad_geo->inv_A_xz * b[it_state].x + grad_geo->inv_A_yz * b[it_state].y + grad_geo->inv_A_zz * b[it_state].z;
    }
  }

  lane_barrier();
  profiler_end_function();
}

#if 0
function void fl_solver_compute_limiter_venkatakrishnan_range(FL_Solver_Euler *euler, FL_State *state, FL_Gradient_State *grad, FL_Limiter_State *limiter, F32 K, Range1_U64 range) {
  profiler_begin_function();

  U64 range_len = range1_u64_len(range);
  UG_Mesh *mesh = euler->mesh;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64            it_cell  = range.min + it_range;
    UG_Cell_Faces *faces    = &mesh->cells.faces[it_cell];
    F32            volume   = mesh->cells.volume[it_cell];
    F32            eps2     = (K * K * K) * volume;

    // NOTE(cmat): Get primitive state
    V5F primitive = {
      .x1 = state->rho[it_cell],
      .x2 = euler->primitive_v_x[it_cell],
      .x3 = euler->primitive_v_y[it_cell],
      .x4 = euler->primitive_v_z[it_cell],
      .x5 = euler->primitive_pressure[it_cell],
    };

    // NOTE(cmat): Initialize min/max.
    V5F primitive_min = { };
    V5F primitive_max = { };
    for Iter_Index(it_state, 5) {
      primitive_min.dat[it_state] = primitive.dat[it_state];
      primitive_max.dat[it_state] = primitive.dat[it_state];
    }

    // NOTE(cmat): Update min/max based on neighbour state values.
    for Iter_Index(it_face, 4) {
      U32 adjacent = faces->adjacent[it_face];
      V5F primitive_adjacent = {
        .x1 = state->rho[adjacent],
        .x2 = euler->primitive_v_x[adjacent],
        .x3 = euler->primitive_v_y[adjacent],
        .x4 = euler->primitive_v_z[adjacent],
        .x5 = euler->primitive_pressure[adjacent],
      };

      for Iter_Index(it_state, 5) {
        primitive_min.dat[it_state] = f32_min(primitive_min.dat[it_state], primitive_adjacent.dat[it_state]);
        primitive_max.dat[it_state] = f32_max(primitive_max.dat[it_state], primitive_adjacent.dat[it_state]);
      }
    }

    // NOTE(cmat): Now that we have the min/max, we apply the venkatakrishnan polynomial expression,
    // - in order to compute phi for each cell.
    for Iter_Index(it_state, 5) {
      V3F cell_grad = v3f(grad->states[it_state].grad_x[it_cell], grad->states[it_state].grad_y[it_cell], grad->states[it_state].grad_z[it_cell]);
      F32 delta_max = primitive_max.dat[it_state] - primitive.dat[it_state];
      F32 delta_min = primitive_min.dat[it_state] - primitive.dat[it_state];
      F32 phi_cell  = 1.f;

      for Iter_Index(it_face, 4) {
        V3F face_center  = v3f(faces->center_x[it_face], faces->center_y[it_face], faces->center_z[it_face]);
        V3F center_delta = v3f_sub(face_center, mesh->cells.center[it_cell]);
        F32 delta_face   = v3f_dot(cell_grad, center_delta);
        F32 phi_face     = 1.f;
        F32 vk_epsilon   = 1e-12f;
        if (delta_face > vk_epsilon || delta_face < -vk_epsilon) {
          F32 delta_bound     = (delta_face > 0.f) ? delta_max : delta_min;
          F32 delta_face_rcp  = 1.f / delta_face;
          F32 y               = delta_bound * delta_face_rcp;
          F32 eps2_norm       = eps2 * delta_face_rcp * delta_face_rcp;
          phi_face            = (y * y + 2.f * y + eps2_norm) / (y * y + y + 2.f + eps2_norm);
        }

        phi_cell = f32_min(phi_cell, phi_face);
      }

      // NOTE(cmat): Assign phi value to cell.
      euler->limiter.states[it_state][it_cell] = phi_cell;
    }
  }

  lane_barrier();
  profiler_end_function();
}

#else

function void fl_solver_compute_limiter_venkatakrishnan_range(FL_Solver_Euler *euler, FL_State *state, FL_Gradient_State *grad, FL_Limiter_State *limiter, F32 K, Range1_U64 range) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);
  UG_Mesh *mesh = euler->mesh;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64            it_cell  = range.min + it_range;
    UG_Cell_Faces *faces    = &mesh->cells.faces[it_cell];
    F32            volume   = mesh->cells.volume[it_cell];
    F32            eps2     = (K * K * K) * volume;
    // NOTE(cmat): Get primitive state
    V5F primitive = {
      .x1 = state->rho[it_cell],
      .x2 = euler->primitive_v_x[it_cell],
      .x3 = euler->primitive_v_y[it_cell],
      .x4 = euler->primitive_v_z[it_cell],
      .x5 = euler->primitive_pressure[it_cell],
    };
    // NOTE(cmat): Initialize min/max.
    V5F primitive_min = { };
    V5F primitive_max = { };
    for Iter_Index(it_state, 5) {
      primitive_min.dat[it_state] = primitive.dat[it_state];
      primitive_max.dat[it_state] = primitive.dat[it_state];
    }
    // NOTE(cmat): Update min/max based on neighbour state values. (unchanged — gather-bound, not vectorized)
    for Iter_Index(it_face, 4) {
      U32 adjacent = faces->adjacent[it_face];
      V5F primitive_adjacent = {
        .x1 = state->rho[adjacent],
        .x2 = euler->primitive_v_x[adjacent],
        .x3 = euler->primitive_v_y[adjacent],
        .x4 = euler->primitive_v_z[adjacent],
        .x5 = euler->primitive_pressure[adjacent],
      };
      for Iter_Index(it_state, 5) {
        primitive_min.dat[it_state] = f32_min(primitive_min.dat[it_state], primitive_adjacent.dat[it_state]);
        primitive_max.dat[it_state] = f32_max(primitive_max.dat[it_state], primitive_adjacent.dat[it_state]);
      }
    }
    // NOTE(cmat): Now that we have the min/max, we apply the venkatakrishnan polynomial expression,
    // - vectorized across the 4 faces, in order to compute phi for each cell.
    F32_X04 zero_v = f32_x04_load_f32(0.f);
    F32_X04 one_v  = f32_x04_load_f32(1.f);
    F32_X04 two_v  = f32_x04_load_f32(2.f);
    F32_X04 eps_v  = f32_x04_load_f32(1e-12f);

    for Iter_Index(it_state, 5) {
      V3F cell_grad  = v3f(grad->states[it_state].grad_x[it_cell], grad->states[it_state].grad_y[it_cell], grad->states[it_state].grad_z[it_cell]);
      F32 delta_max  = primitive_max.dat[it_state] - primitive.dat[it_state];
      F32 delta_min  = primitive_min.dat[it_state] - primitive.dat[it_state];

      F32_X04 face_center_x = f32_x04_load(faces->center_x);
      F32_X04 face_center_y = f32_x04_load(faces->center_y);
      F32_X04 face_center_z = f32_x04_load(faces->center_z);

      F32_X04 cell_center_x = f32_x04_load_f32(mesh->cells.center[it_cell].x);
      F32_X04 cell_center_y = f32_x04_load_f32(mesh->cells.center[it_cell].y);
      F32_X04 cell_center_z = f32_x04_load_f32(mesh->cells.center[it_cell].z);

      F32_X04 delta_x = f32_x04_sub(face_center_x, cell_center_x);
      F32_X04 delta_y = f32_x04_sub(face_center_y, cell_center_y);
      F32_X04 delta_z = f32_x04_sub(face_center_z, cell_center_z);

      F32_X04 grad_x_v = f32_x04_load_f32(cell_grad.x);
      F32_X04 grad_y_v = f32_x04_load_f32(cell_grad.y);
      F32_X04 grad_z_v = f32_x04_load_f32(cell_grad.z);

      // NOTE(cmat): delta_face for all 4 faces at once — this is the dot(cell_grad, center_delta) from the scalar version.
      F32_X04 delta_face = f32_x04_mul(grad_x_v, delta_x);
      delta_face         = f32_x04_fused_mul_add(grad_y_v, delta_y, delta_face);
      delta_face         = f32_x04_fused_mul_add(grad_z_v, delta_z, delta_face);

      F32_X04 delta_max_v = f32_x04_load_f32(delta_max);
      F32_X04 delta_min_v = f32_x04_load_f32(delta_min);
      F32_X04 eps2_v      = f32_x04_load_f32(eps2);

      // NOTE(cmat): abs(delta_face) > vk_epsilon, computed branchlessly for all 4 faces.
      F32_X04 abs_delta_face = f32_x04_max(delta_face, f32_x04_sub(zero_v, delta_face));
      F32_X04 valid_mask     = f32_x04_cmp_gt(abs_delta_face, eps_v);

      // NOTE(cmat): delta_bound = delta_face > 0 ? delta_max : delta_min
      F32_X04 positive_mask = f32_x04_cmp_gt(delta_face, zero_v);
      F32_X04 delta_bound   = f32_x04_select(positive_mask, delta_max_v, delta_min_v);

      // NOTE(cmat): Safe to divide even where delta_face ~ 0 — result may be Inf/NaN there,
      // - but valid_mask discards that lane in the final select below (bit-select, not arithmetic).
      F32_X04 delta_face_rcp = f32_x04_div(one_v, delta_face);
      F32_X04 y              = f32_x04_mul(delta_bound, delta_face_rcp);
      F32_X04 eps2_norm      = f32_x04_mul(f32_x04_mul(eps2_v, delta_face_rcp), delta_face_rcp);

      F32_X04 y2          = f32_x04_mul(y, y);
      F32_X04 numerator   = f32_x04_add(f32_x04_fused_mul_add(two_v, y, y2), eps2_norm);
      F32_X04 denominator = f32_x04_add(f32_x04_add(y2, y), f32_x04_add(two_v, eps2_norm));
      F32_X04 phi_computed = f32_x04_div(numerator, denominator);

      F32_X04 phi_face_v = f32_x04_select(valid_mask, phi_computed, one_v);

      // NOTE(cmat): Horizontal min across the 4 faces -> phi_cell for this state.
      F32 phi_lanes[4];
      f32_x04_store(phi_lanes, phi_face_v);
      F32 phi_cell = f32_min(f32_min(phi_lanes[0], phi_lanes[1]), f32_min(phi_lanes[2], phi_lanes[3]));

      // NOTE(cmat): Assign phi value to cell.
      euler->limiter.states[it_state][it_cell] = phi_cell;
    }
  }
  lane_barrier();
  profiler_end_function();
}


#endif


function F32 fl_solver_compute_global_time_step(FL_Solver_Euler *euler, F64 *time_steps) {
  profiler_begin_function();

  // NOTE(cmat): Compute minimum time step for each lane.
  F64 lane_time_step = f64_limit_max;
  for Iter_Range(it, lane_range(euler->mesh->cells.len)) {
    lane_time_step = f64_min(lane_time_step, time_steps[it]);
  }

  euler->lane_time_step[lane_index()] = lane_time_step;

  // NOTE(cmat): Compute minimum time step across lanes.
  lane_barrier();

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

function V3_F64 fl_solver_euler_compute_state_norm2(FL_Solver_Euler *euler, FL_State *state, Range1_U64 range) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);
  UG_Mesh *mesh = euler->mesh;

  F64 density_norm2 = 0, momentum_norm2 = 0, energy_norm2 = 0;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64 it_cell      = range.min + it_range;
    F32 rho          = state->rho[it_cell];
    density_norm2   += (F64)rho * rho;
    V3F velocity     = v3f(state->rho_v1[it_cell], state->rho_v2[it_cell], state->rho_v3[it_cell]);
    momentum_norm2  += v3f_len2(velocity);
    F32 e            = state->energy[it_cell];
    energy_norm2    += (F64)e * e;
  }

  euler->lane_state_norm2[lane_index()] = v3_f64(density_norm2, momentum_norm2, energy_norm2);
  lane_barrier();

  V3_F64 state_norm2 = v3_f64(0, 0, 0);
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      state_norm2 = v3_f64_add(state_norm2, euler->lane_state_norm2[it]);
    }
  }

  lane_broadcast_type(&state_norm2, 0);
  state_norm2.x = ipc_rank_sum_f64(state_norm2.x);
  state_norm2.y = ipc_rank_sum_f64(state_norm2.y);
  state_norm2.z = ipc_rank_sum_f64(state_norm2.z);

  lane_barrier();
  profiler_end_function();
  return state_norm2;
}

function void fl_solver_euler_compute_residual(FL_Solver_Euler *euler, FL_State *state, FL_State *residual, B32 compute_time_step) {
  profiler_begin_function();
  UG_Mesh *mesh = euler->mesh;

  // NOTE(cmat): Pack cells for halo exchange.
  lane_barrier();
  fl_solver_euler_halo_state_pack_send_data(euler, state);

  // NOTE(cmat): Start halo cell state exchange between partitions.
  IPC_Request_Scope(&euler->halo_state_request_list) {

    // NOTE(cmat): Compute primitive variables for local cells.
    fl_solver_euler_compute_primitive_range(euler, state, range1_u64(0, mesh->cells.len));
    
    // NOTE(cmat): While we are waiting for the halo cells to arrive from,
    // - other ranks, we compute the ghost cells to save time.
    fl_solver_compute_ghost(euler, state);

    // NOTE(cmat): Compute primitive variables for ghost cells.
    fl_solver_euler_compute_primitive_range(euler, state, range1_u64(mesh->cells.len + mesh->halos.len, mesh->cells.len + mesh->halos.len + mesh->ghosts.len));

    // NOTE(cmat): Next, we compute gradients and limiters for each local cell.
    // - Those are cells not touching any halo cells; they can still be in touch with ghost cells.
    fl_solver_compute_gradient_range                (euler, state, &euler->gradient, mesh->groups.cells_interior);
    fl_solver_compute_limiter_venkatakrishnan_range (euler, state, &euler->gradient, &euler->limiter, 3.f, mesh->groups.cells_interior);
  }

  // NOTE(cmat): Unpack received halo state data.
  fl_solver_euler_halo_state_unpack_receive_data(euler, state);

  // NOTE(cmat): Compute primitive variables for halo cells.
  fl_solver_euler_compute_primitive_range(euler, state, range1_u64(mesh->cells.len, mesh->cells.len + mesh->halos.len));

  // NOTE(cmat): Compute gradients and limiters for remaining boundary cells
  fl_solver_compute_gradient_range                (euler, state, &euler->gradient, mesh->groups.cells_boundary);
  fl_solver_compute_limiter_venkatakrishnan_range (euler, state, &euler->gradient, &euler->limiter, 3.f, mesh->groups.cells_boundary);

  // NOTE(cmat): Pack cells for gradient exchange.
  fl_solver_euler_halo_gradient_limiter_pack_send_data(euler, &euler->gradient, &euler->limiter);

  // NOTE(cmat): Start halo cell gradient exchange between partitions.
  IPC_Request_Scope(&euler->halo_gradient_limiter_request_list) {

    // NOTE(cmat): Compute the residual of all interior cells.
    fl_solver_compute_residual_range(euler, state, residual, &euler->gradient, mesh->groups.cells_interior, compute_time_step, euler->cell_time_step);
  }

  // NOTE(cmat): Unpack received gradient state data.
  fl_solver_euler_halo_gradient_limiter_unpack_receive_data(euler, &euler->gradient, &euler->limiter);

  // NOTE(cmat): Compute residual for remaining boundary cells
  fl_solver_compute_residual_range(euler, state, residual, &euler->gradient, mesh->groups.cells_boundary, compute_time_step, euler->cell_time_step);

  lane_barrier();
  profiler_end_function();
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
      state_dst_array[it] = (F32)(state_1_coeff * state_1_array[it] + state_2_coeff * state_2_array[it] + scaled_time_step * residual_array[it]);
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



function F64 fl_solver_euler_solve_global_step_forward_euler(FL_Solver_Euler *euler, F32 CFL) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  // NOTE(cmat): Compute residual.
  fl_solver_euler_compute_residual(euler, &euler->flow_1, &euler->residual, 1);

  // NOTE(cmat): Global time-stepping.
  F64 time_step = fl_solver_compute_global_time_step(euler, euler->cell_time_step);
  time_step *= CFL;

  fl_solver_global_euler_step(euler, &euler->flow_1, &euler->flow_1, &euler->residual, 1.f, time_step);
  
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

  // NOTE(cmat): Scale CFL by 2.
  CFL *= 2;

  // NOTE(cmat): Compute R(Q1)
  fl_solver_euler_compute_residual(euler, &euler->flow_1, &euler->residual, 1);

  // NOTE(cmat): Compute time-step, based on R(Q1)
  F64 time_step = fl_solver_compute_global_time_step(euler, euler->cell_time_step);
  time_step *= CFL;

  // NOTE(cmat): Compute Q2 = Q1 + dt/2 * R(Q1)
  fl_solver_global_euler_step(euler, &euler->flow_2, &euler->flow_1, &euler->residual, .5f, time_step);

  // NOTE(cmat): Compute R(Q2)
  fl_solver_euler_compute_residual(euler, &euler->flow_2, &euler->residual, 0);

  // NOTE(cmat): Compute Q2 = Q2 + dt / 2 * R(Q2)
  fl_solver_global_euler_step(euler, &euler->flow_2, &euler->flow_2, &euler->residual, .5f, time_step);

  // NOTE(cmat): Compute R(Q2)
  fl_solver_euler_compute_residual(euler, &euler->flow_2, &euler->residual, 0);

  // NOTE(cmat): Compute  Q2 = 2/3 * Q1 + 1/3 * [ Q2 + dt/2 * R(Q2) ]
  //                      Q2 = 2/3 * Q1 + 1/3 * Q2 + dt/2 * R(Q2)
  fl_solver_global_euler_step_2(euler, &euler->flow_2, 2.f/3.f, &euler->flow_1, 1.f/3.f, &euler->flow_2, &euler->residual, 1.f / 6.f, time_step);

  // NOTE(cmat): Compute R(Q2)
  fl_solver_euler_compute_residual(euler, &euler->flow_2, &euler->residual, 0);

  // U = Q2 + dt/2 * R(Q2)
  fl_solver_global_euler_step(euler, &euler->flow_1, &euler->flow_2, &euler->residual, .5f, time_step);
  
  profiler_end_function();
  return time_step;
}

function void fl_solver_euler_solve_local_step_SSP_RK_4_3(FL_Solver_Euler *euler, F32 CFL) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  // NOTE(cmat): Scale CFL by 2.
  CFL *= 2;

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

function F32 fl_solver_euler_solve(FL_Solver_Euler *euler, F32 time_target) {
  profiler_begin_function();
  log_zone_start("Solving euler flow");

  // NOTE(cmat): Synchronize all ranks, for more accurate benchmarking.
  ipc_rank_barrier();

  F32 CFL = 0.85f;
  U64 clock_start = sys_performance_clock_now();

  // NOTE(cmat): Iterate.
  F64 time        = 0;
  U64 iteration   = 0;

  static B32    residual_norm_init  = 0;
  static V3_F64 residual_norm_first = { 0, 0, 0 };

  for Iter_Index(it, 10000) {
  // for Iter_Index(it, 64) {
  // while (time < .2f) {
    // fl_solver_euler_solve_local_step_forward_euler(euler, CFL);
    // fl_solver_euler_solve_local_step_SSP_RK_4_3(euler, CFL);

#if 1
    F64 time_step = fl_solver_euler_solve_global_step_SSP_RK_4_3(euler, CFL);
#else
    fl_solver_euler_solve_local_step_SSP_RK_4_3(euler, CFL);
    F64 time_step = 0;
#endif

    time         += time_step;
    iteration    += 1;

#if 1
    // if (!residual_norm_init || it == 9999) {
    if (1) {

      // NOTE(cmat): Compute current residual.
      fl_solver_euler_compute_residual(euler, &euler->flow_1, &euler->residual, 0);
      
      // NOTE(cmat): Compute residual norm.
      V3_F64 residual_norm = fl_solver_euler_compute_state_norm2(euler, &euler->residual, range1_u64(0, euler->mesh->cells.len));

      residual_norm   = v3_f64_div  (residual_norm, (F64)euler->mesh->cells.len);
      residual_norm.x = f64_sqrt    (residual_norm.x);
      residual_norm.y = f64_sqrt    (residual_norm.y);
      residual_norm.z = f64_sqrt    (residual_norm.z);
      
      If_Unlikely (!residual_norm_init) {
        residual_norm_init = 1;
        residual_norm_first = residual_norm;
      }

      residual_norm.x /= residual_norm_first.x;
      residual_norm.y /= residual_norm_first.y;
      residual_norm.z /= residual_norm_first.z;

      log_info("TIME %.2g | TIMESTEP %.2g | ITERATION %'llu | RESIDUAL %.2g, %.2g, %.2g", time, time_step, iteration, residual_norm.x, residual_norm.y, residual_norm.z);
    }
#endif
  }

  lane_barrier();

  U64 clock_end       = sys_performance_clock_now();
  U64 clock_dt        = clock_end - clock_start;
  F64 clock_seconds   = clock_dt * sys_performance_clock_to_nanoseconds() * 1e-9;

  log_info("Simulation time: %.4f seconds", time);
  log_info("Wall-Clock time: %.4f seconds", clock_seconds);

  log_zone_end();
  profiler_end_function();

  return time;
}
