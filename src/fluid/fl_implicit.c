#define NEWTON_GMRES_TOL     1e-2f  // relative — Newton only needs an inexact linear solve
#define NEWTON_MAX_ITERS     10
#define NEWTON_TOL           1e-2f  // relative drop in ||N(Q)|| to accept a Newton step

// ------------------------------------------------------------
// #-- FL_State as vector helpers

function void fl_state_zero(FL_State *a, Range1_U64 range) {
  U64 range_len = range1_u64_len(range);
  for Iter_Index(it_state, 5) {
    F32 *a_arr = a->states[it_state];
    for Iter_Range(it_range, lane_range(range_len)) {
      U64 it = range.min + it_range;
      a_arr[it] = 0.f;
    }
  }
  lane_barrier();
}

function void fl_state_copy(FL_State *dst, FL_State *src, Range1_U64 range) {
  U64 range_len = range1_u64_len(range);
  for Iter_Index(it_state, 5) {
    F32 *d_arr = dst->states[it_state];
    F32 *s_arr = src->states[it_state];
    for Iter_Range(it_range, lane_range(range_len)) {
      U64 it = range.min + it_range;
      d_arr[it] = s_arr[it];
    }
  }
  lane_barrier();
}

function void fl_state_copy_scaled(FL_State *dst, FL_State *src, F32 scale, Range1_U64 range) {
  U64 range_len = range1_u64_len(range);
  for Iter_Index(it_state, 5) {
    F32 *d_arr = dst->states[it_state];
    F32 *s_arr = src->states[it_state];
    for Iter_Range(it_range, lane_range(range_len)) {
      U64 it = range.min + it_range;
      d_arr[it] = s_arr[it] * scale;
    }
  }
  lane_barrier();
}

// dst = alpha*dst + beta*src   (matches "axpy" call sites used below)
function void fl_state_axpy_in_place(FL_State *dst, F32 alpha, F32 beta, FL_State *src, Range1_U64 range) {
  U64 range_len = range1_u64_len(range);
  for Iter_Index(it_state, 5) {
    F32 *d_arr = dst->states[it_state];
    F32 *s_arr = src->states[it_state];
    for Iter_Range(it_range, lane_range(range_len)) {
      U64 it = range.min + it_range;
      d_arr[it] = alpha * d_arr[it] + beta * s_arr[it];
    }
  }
  lane_barrier();
}

function void fl_state_scale_in_place(FL_State *dst, F32 scale, Range1_U64 range) {
  U64 range_len = range1_u64_len(range);
  for Iter_Index(it_state, 5) {
    F32 *d_arr = dst->states[it_state];
    for Iter_Range(it_range, lane_range(range_len)) {
      U64 it = range.min + it_range;
      d_arr[it] *= scale;
    }
  }
  lane_barrier();
}


// ------------------------------------------------------------
// #-- Global Reductions

function F32 fl_solver_euler_global_dot(FL_Solver_Euler *euler, FL_State *a, FL_State *b, Range1_U64 range) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);

  F32 local_dot = 0.f;
  for Iter_Index(it_state, 5) {
    F32 *a_arr = a->states[it_state];
    F32 *b_arr = b->states[it_state];
    for Iter_Range(it_range, lane_range(range_len)) {
      U64 it = range.min + it_range;
      local_dot += (F32)a_arr[it] * b_arr[it];
    }
  }

  euler->reduce_scratch[lane_index()] = local_dot;
  lane_barrier();

  F32 global_dot = 0.f;
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      global_dot += euler->reduce_scratch[it];
    }
  }
  lane_broadcast_type(&global_dot, 0);
  global_dot = (F32)ipc_rank_sum_f64((F64)global_dot);

  lane_barrier();
  profiler_end_function();
  return global_dot;
}

function F32 fl_solver_euler_global_norm(FL_Solver_Euler *euler, FL_State *a, Range1_U64 range) {
  return f32_sqrt(fl_solver_euler_global_dot(euler, a, a, range));
}

// Reduce a single scalar (already local-min'd within this lane's slice) to a global minimum
// across lanes and IPC ranks. Used by the physicality step limiter.
function F32 fl_solver_euler_global_min(FL_Solver_Euler *euler, F32 lane_local_min) {
  profiler_begin_function();

  euler->reduce_scratch[lane_index()] = lane_local_min;
  lane_barrier();

  F32 global_min = f32_limit_max;
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      global_min = f32_min(global_min, euler->reduce_scratch[it]);
    }
  }
  lane_broadcast_type(&global_min, 0);
  global_min = ipc_rank_minimum_f32(global_min);

  lane_barrier();
  profiler_end_function();
  return global_min;
}

// ------------------------------------------------------------
// #-- Frozen Gradient Resiudal

function void fl_solver_euler_compute_residual_frozen(FL_Solver_Euler *euler, FL_State *state, FL_State *residual) {
  profiler_begin_function();
  UG_Mesh *mesh = euler->mesh;

  lane_barrier();
  fl_solver_euler_halo_state_pack_send_data(euler, state);

  IPC_Request_Scope(&euler->halo_state_request_list) {
    fl_solver_euler_compute_primitive_range(euler, state, range1_u64(0, mesh->cells.len));
    fl_solver_compute_ghost(euler, state);
    fl_solver_euler_compute_primitive_range(euler, state,
        range1_u64(mesh->cells.len + mesh->halos.len,
                   mesh->cells.len + mesh->halos.len + mesh->ghosts.len));
  }

  fl_solver_euler_halo_state_unpack_receive_data(euler, state);
  fl_solver_euler_compute_primitive_range(euler, state, range1_u64(mesh->cells.len, mesh->cells.len + mesh->halos.len));

  // NOTE(cmat/jfnk): euler->gradient / euler->limiter are intentionally NOT recomputed or
  // re-exchanged here — they hold the values from the unperturbed Q at the current Newton
  // iterate. This is the "frozen" approximation: correct for a Jacobian-vector product,
  // not a substitute for the true nonlinear residual used in the Newton convergence check.
  fl_solver_compute_residual_range(euler, state, residual, &euler->gradient, mesh->groups.cells_interior, 0, 0);
  fl_solver_compute_residual_range(euler, state, residual, &euler->gradient, mesh->groups.cells_boundary, 0, 0);

  lane_barrier();
  profiler_end_function();
}


// ------------------------------------------------------------
// #-- Matrix Free Jacobian-Vector Product and Preconditioner, Global Time-Stepping.

function void fl_solver_euler_jacobian_vector_product_global(FL_Solver_Euler *euler, FL_State *Q0, FL_State *R0, FL_State *v, FL_State *Jv_out) {
  profiler_begin_function();
  UG_Mesh *mesh = euler->mesh;
  Range1_U64 range = range1_u64(0, mesh->cells.len);

  F32 v_norm = fl_solver_euler_global_norm(euler, v, range);
  F32 q_norm = fl_solver_euler_global_norm(euler, Q0, range);
  F32 eps    = f32_sqrt(f32_limit_epsilon) * (1.f + q_norm) / f32_max(v_norm, 1e-30f);

  for Iter_Index(it_state, 5) {
    F32 *q0_arr = Q0->states[it_state];
    F32 *v_arr  = v->states[it_state];
    F32 *qp_arr = euler->newton_state_pert.states[it_state];
    for Iter_Range(it, lane_range(mesh->cells.len)) {
      qp_arr[it] = q0_arr[it] + eps * v_arr[it];
    }
  }
  lane_barrier();

  fl_solver_euler_compute_residual_frozen(euler, &euler->newton_state_pert, &euler->newton_residual_pert);

  for Iter_Index(it_state, 5) {
    F32 *r0_arr = R0->states[it_state];
    F32 *rp_arr = euler->newton_residual_pert.states[it_state];
    F32 *jv_arr = Jv_out->states[it_state];
    for Iter_Range(it, lane_range(mesh->cells.len)) {
      jv_arr[it] = (rp_arr[it] - r0_arr[it]) / eps;
    }
  }
  lane_barrier();
  profiler_end_function();
}

// M^-1 r ~= r / (1/dt + spectral_radius_sum / volume), single global dt.
function void fl_solver_euler_apply_preconditioner_global(FL_Solver_Euler *euler, FL_State *r, FL_State *z, F32 time_coeff) {
  UG_Mesh *mesh = euler->mesh;
  for Iter_Index(it_state, 5) {
    F32 *r_arr = r->states[it_state];
    F32 *z_arr = z->states[it_state];
    for Iter_Range(it, lane_range(mesh->cells.len)) {
      F32 volume   = mesh->cells.volume[it];
      F32 spectral = euler->cell_spectral_inviscid_sum[it] + euler->cell_spectral_viscous_sum[it];
      F32 diag     = time_coeff + spectral / volume;
      z_arr[it]    = r_arr[it] / diag;
    }
  }
  lane_barrier();
}

// ------------------------------------------------------------
// #-- GMRES helper

// H is row-major with row stride NEWTON_GMRES_M. Applies existing Givens rotations to column
// j, computes and applies the new rotation eliminating H[j+1][j], updates g. Returns the
// resulting (unreduced) residual estimate |g[j+1]|.
function F32 gmres_apply_givens_and_update(F32 *H, F32 *cs, F32 *sn, F32 *g, U32 j) {
  for Iter_Index(i, j) {
    F32 h_i  = H[i * NEWTON_GMRES_M + j];
    F32 h_i1 = H[(i + 1) * NEWTON_GMRES_M + j];
    H[i * NEWTON_GMRES_M + j]       =  cs[i] * h_i + sn[i] * h_i1;
    H[(i + 1) * NEWTON_GMRES_M + j] = -sn[i] * h_i + cs[i] * h_i1;
  }

  F32 h_jj  = H[j * NEWTON_GMRES_M + j];
  F32 h_j1j = H[(j + 1) * NEWTON_GMRES_M + j];
  F32 denom = f32_sqrt(h_jj * h_jj + h_j1j * h_j1j);

  F32 c = 1.f, s = 0.f;
  if (denom > 1e-30f) {
    c = h_jj  / denom;
    s = h_j1j / denom;
  }
  cs[j] = c;
  sn[j] = s;

  H[j * NEWTON_GMRES_M + j]       = c * h_jj + s * h_j1j;
  H[(j + 1) * NEWTON_GMRES_M + j] = 0.f;

  F32 g_j  = g[j];
  F32 g_j1 = g[j + 1];
  g[j]     =  c * g_j + s * g_j1;
  g[j + 1] = -s * g_j + c * g_j1;

  return f32_abs(g[j + 1]);
}

// Back-substitution for the (m_used x m_used) upper-triangular system H*y = g.
function void gmres_back_substitute(F32 *H, F32 *g, U32 m_used, F32 *y) {
  for (I32 i = (I32)m_used - 1; i >= 0; i -= 1) {
    F32 sum = g[i];
    for (U32 k = i + 1; k < m_used; k += 1) {
      sum -= H[i * NEWTON_GMRES_M + k] * y[k];
    }
    y[i] = sum / H[i * NEWTON_GMRES_M + i];
  }
}

// ------------------------------------------------------------
// #-- GMRES(m), matrix-free, right-preconditioned, global time-stepping

// Solves A*dQ = b where A(v) = v/dt - Jv(v), right-preconditioned by M^-1 (Section 7).
// Writes the solution into dQ. Returns the achieved relative residual.
function F32 fl_solver_euler_gmres_solve_global(FL_Solver_Euler *euler, FL_State *Q0, FL_State *R0, FL_State *b, F32 time_coeff, F32 gmres_tol, FL_State *dQ) {
  profiler_begin_function();
  UG_Mesh *mesh = euler->mesh;
  Range1_U64 range = range1_u64(0, mesh->cells.len);

  F32 beta = fl_solver_euler_global_norm(euler, b, range);
  if (beta < 1e-30f) {
    fl_state_zero(dQ, range);
    profiler_end_function();
    return 0.f;
  }

  fl_state_copy_scaled(&euler->krylov_basis[0], b, 1.f / beta, range);
  // Zero_Fill_Count(euler->gmres_g, NEWTON_GMRES_M + 1);
  // Stack_Array_Zero(euler->gmres_g);
  for Iter_Index(it, NEWTON_GMRES_M + 1) {
    euler->gmres_g[it] = 0;
  }

  euler->gmres_g[0] = beta;

  U32 m_used = NEWTON_GMRES_M;
  for Iter_Index(j, NEWTON_GMRES_M) {

    // z = M^-1 V_j  (right preconditioning)
    fl_solver_euler_apply_preconditioner_global(euler, &euler->krylov_basis[j], &euler->krylov_z, time_coeff);
    fl_solver_euler_jacobian_vector_product_global(euler, Q0, R0, &euler->krylov_z, &euler->krylov_basis[j + 1]);
    fl_state_axpy_in_place(&euler->krylov_basis[j + 1], -1.f, time_coeff, &euler->krylov_z, range);   // CHANGED: was 1.f/dt

    // Modified Gram-Schmidt against V_0..V_j
    for Iter_Index(i, j + 1) {
      F32 h_ij = fl_solver_euler_global_dot(euler, &euler->krylov_basis[j + 1], &euler->krylov_basis[i], range);
      euler->hessenberg[i * NEWTON_GMRES_M + j] = h_ij;
      fl_state_axpy_in_place(&euler->krylov_basis[j + 1], 1.f, -h_ij, &euler->krylov_basis[i], range);
    }

    F32 h_next = fl_solver_euler_global_norm(euler, &euler->krylov_basis[j + 1], range);
    euler->hessenberg[(j + 1) * NEWTON_GMRES_M + j] = h_next;

    if (h_next > 1e-30f) {
      fl_state_scale_in_place(&euler->krylov_basis[j + 1], 1.f / h_next, range);
    }

    F32 residual_estimate = gmres_apply_givens_and_update(euler->hessenberg, euler->givens_cs, euler->givens_sn, euler->gmres_g, j);

#if 0
    if (residual_estimate / beta < NEWTON_GMRES_TOL || h_next < 1e-30f) {
      m_used = j + 1;
      break;
    }
#else
    if (residual_estimate / beta < gmres_tol || h_next < 1e-30f) {
      m_used = j + 1;
      break;
    }
#endif
  }

  F32 y[NEWTON_GMRES_M];
  gmres_back_substitute(euler->hessenberg, euler->gmres_g, m_used, y);

  fl_state_zero(dQ, range);
  for Iter_Index(i, m_used) {
    fl_solver_euler_apply_preconditioner_global(euler, &euler->krylov_basis[i], &euler->krylov_z, time_coeff);
    fl_state_axpy_in_place(dQ, 1.f, y[i], &euler->krylov_z, range);
  }

#if 0
  F32 relative_residual = euler->gmres_g[m_used] / beta;
#else
  F32 relative_residual = f32_abs(euler->gmres_g[m_used] / beta);
#endif
 

  // TODO(cmat): NEW, testing.
  if (lane_index() == 0) {
    log_info("    GMRES: %u/%u iters | rel_residual %.3g", m_used, NEWTON_GMRES_M, relative_residual);
  }

  profiler_end_function();
  return relative_residual;
}


// ------------------------------------------------------------
// #-- Physically limited Newton step length

function F32 fl_solver_euler_compute_max_physical_step(FL_Solver_Euler *euler, FL_State *Q, FL_State *dQ, Range1_U64 range) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);

  F32 lane_alpha = 1.f;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64 it = range.min + it_range;

    // NOTE(cmat): density floor.
    F32 rho   = Q->rho[it];
    F32 d_rho = dQ->rho[it];
    if (d_rho < 0.f) {
      lane_alpha = f32_min(lane_alpha, .95f * (-rho / d_rho));
    }

    // NOTE(cmat): pressure floor. Directional derivative of p(Q + alpha*dQ) at alpha=0,
    // estimated with a cheap local finite difference (no need for the full primitive pass).
    F32 rho1 = Q->rho[it], rv1 = Q->rho_v1[it], rv2 = Q->rho_v2[it], rv3 = Q->rho_v3[it], e = Q->energy[it];
    F32 rho_rcp = 1.f / rho1;
    F32 ke = .5f * (rv1*rv1 + rv2*rv2 + rv3*rv3) * rho_rcp;
    F32 p  = (Q->material.gamma - 1.f) * (e - ke);

    F32 eps_fd = 1e-6f * f32_max(f32_abs(e), 1.f);
    F32 rho2  = rho1 + eps_fd * dQ->rho[it];
    F32 rv1b  = rv1  + eps_fd * dQ->rho_v1[it];
    F32 rv2b  = rv2  + eps_fd * dQ->rho_v2[it];
    F32 rv3b  = rv3  + eps_fd * dQ->rho_v3[it];
    F32 eb    = e    + eps_fd * dQ->energy[it];
    F32 ke2   = .5f * (rv1b*rv1b + rv2b*rv2b + rv3b*rv3b) / rho2;
    F32 p2    = (Q->material.gamma - 1.f) * (eb - ke2);
    F32 dp_dalpha = (p2 - p) / eps_fd;

    if (dp_dalpha < 0.f) {
      lane_alpha = f32_min(lane_alpha, .95f * (-p / dp_dalpha));
    }
  }
  lane_barrier();

  F32 alpha = fl_solver_euler_global_min(euler, lane_alpha);
  profiler_end_function();
  return f32_clamp(alpha, 1e-4f, 1.f);
}

// ------------------------------------------------------------
// #-- Global backward-euler JFNK step

function F32 fl_solver_euler_solve_global_step_backward_euler_BDF2_JFNK(FL_Solver_Euler *euler, F32 CFL) {
  profiler_begin_function();
  UG_Mesh *mesh = euler->mesh;
  Range1_U64 range = range1_u64(0, mesh->cells.len);

  fl_solver_euler_compute_residual(euler, &euler->flow_1, &euler->residual, 1);
  F32 dt = fl_solver_compute_global_time_step(euler, euler->cell_time_step) * CFL;

  // BDF2 time coefficient once we have two prior levels; plain backward Euler
  // (1st order) to bootstrap the very first step, since BDF2 needs Q^{n-1}.
  F32 time_coeff = euler->has_prev_step ? (1.5f / dt) : (1.f / dt);

  // Shift history BEFORE overwriting flow_2 — flow_0 becomes the old Q^n (soon to be Q^{n-1}).
  if (euler->has_prev_step) {
    fl_state_copy(&euler->flow_0, &euler->flow_2, range);
  }

  // Q^n snapshot — reuse flow_2 the same way the RK(4,3) code uses it as a stage buffer.
  fl_state_copy(&euler->flow_2, &euler->flow_1, range);

  F32 n0_norm       = 0.f;
  F32 n_norm        = 0.f;
  F32 prev_ratio    = 1.f;
  U32 stagnant_count = 0;
  U32 k_used        = NEWTON_MAX_ITERS;
  B32 stagnated     = 0;

  for Iter_Index(k, NEWTON_MAX_ITERS) {

    // True nonlinear residual (limiter recomputed) — used for the convergence check and as
    // the b-vector's R(Q^k) term. Also refreshes cell_spectral_*_sum every iteration (fix #2)
    // so the preconditioner stays current with the Newton iterate.
    fl_solver_euler_compute_residual(euler, &euler->flow_1, &euler->newton_residual0, 1);

    // RHS branches on BDF2 vs bootstrap backward Euler.
    if (euler->has_prev_step) {
      for Iter_Index(it_state, 5) {
        F32 *r_arr    = euler->newton_residual0.states[it_state];
        F32 *q_arr    = euler->flow_1.states[it_state];
        F32 *qn_arr   = euler->flow_2.states[it_state];
        F32 *qnm1_arr = euler->flow_0.states[it_state];
        F32 *b_arr    = euler->newton_rhs.states[it_state];
        for Iter_Range(it, lane_range(mesh->cells.len)) {
          b_arr[it] = r_arr[it] - (1.5f * q_arr[it] - 2.f * qn_arr[it] + 0.5f * qnm1_arr[it]) / dt;
        }
      }
    } else {
      for Iter_Index(it_state, 5) {
        F32 *r_arr  = euler->newton_residual0.states[it_state];
        F32 *q_arr  = euler->flow_1.states[it_state];
        F32 *qn_arr = euler->flow_2.states[it_state];
        F32 *b_arr  = euler->newton_rhs.states[it_state];
        for Iter_Range(it, lane_range(mesh->cells.len)) {
          b_arr[it] = r_arr[it] - (q_arr[it] - qn_arr[it]) / dt;
        }
      }
    }
    lane_barrier();

    n_norm = fl_solver_euler_global_norm(euler, &euler->newton_rhs, range);
    if (k == 0) {
      n0_norm = n_norm;
    }

    F32 ratio = n_norm / f32_max(n0_norm, 1e-30f);
    if (ratio < NEWTON_TOL) {
      k_used = k;
      break;
    }

    // Stagnation check — if two consecutive iterations each improve the residual ratio
    // by less than 5%, the Jacobian approximation (F32 finite-difference JVP) has stopped
    // carrying useful information; further iterations just burn GMRES solves for no benefit.
    if (k > 0) {
      F32 improvement = (prev_ratio - ratio) / f32_max(prev_ratio, 1e-30f);
      if (improvement < 0.05f) {
        stagnant_count += 1;
      } else {
        stagnant_count = 0;
      }
      if (stagnant_count >= 2) {
        k_used    = k;
        stagnated = 1;
        break;
      }
    }
    prev_ratio = ratio;

    // Eisenstat-Walker adaptive forcing term. Loose early (fast, cheap GMRES solves while
    // far from the root), tight later (accurate solves needed to actually approach
    // NEWTON_TOL) — floored so we're never looser than a sane cap, and never asked to be
    // more accurate than NEWTON_TOL itself would require anyway.
    F32 gmres_tol = f32_clamp(0.1f * ratio, NEWTON_TOL, 1e-1f);

    if (lane_index() == 0) {
      log_info("    Newton k=%u | forcing gmres_tol=%.2g", k, gmres_tol);
    }

    fl_solver_euler_gmres_solve_global(euler, &euler->flow_1, &euler->newton_residual0, &euler->newton_rhs, time_coeff, gmres_tol, &euler->newton_dQ);

    F32 alpha = fl_solver_euler_compute_max_physical_step(euler, &euler->flow_1, &euler->newton_dQ, range);
    fl_state_axpy_in_place(&euler->flow_1, 1.f, alpha, &euler->newton_dQ, range);
  }

  if (lane_index() == 0) {
    B32 converged = (k_used < NEWTON_MAX_ITERS) && !stagnated;
    const char *status = converged ? "" : (stagnated ? "  *** STAGNATED (Jacobian noise floor?) ***" : "  *** DID NOT CONVERGE ***");
    log_info("  Newton: %u/%u iters | residual ratio %.3g%s",
              k_used, NEWTON_MAX_ITERS,
              n_norm / f32_max(n0_norm, 1e-30f),
              status);
  }

  euler->has_prev_step = 1;   // from here on, every step uses BDF2 

  profiler_end_function();
  return dt;
}


function F32 fl_solver_euler_solve_implicit(FL_Solver_Euler *euler, F32 time_target) {
  profiler_begin_function();
  log_zone_start("Solving euler flow");

  // NOTE(cmat): Synchronize all ranks, for more accurate benchmarking.
  ipc_rank_barrier();

  F32 CFL_max     = 1000.0f;
  F32 CFL_growth  = 1.03f;

  // NOTE(cmat): Starting value.
  static F32 CFL  = 0.1f;

  U64 clock_start = sys_performance_clock_now();

  // NOTE(cmat): Iterate.
  F64 time        = 0;
  U64 iteration   = 0;

  static B32 residual_norm_init  = 0;
  static V3_F64 residual_norm_first = { 0, 0, 0 };

  for Iter_Index(it, 100) {
    F32 time_step = fl_solver_euler_solve_global_step_backward_euler_BDF2_JFNK(euler, CFL);
    time         += time_step;
    iteration    += 1;

    if (lane_index() == 0) {
      CFL = f32_min(CFL_max, CFL * CFL_growth);
    }

    lane_broadcast_type(&CFL, 0);

#if 1
    // if (!residual_norm_init || it == 9999) {
    if (1) {

      // NOTE(cmat): Compute current residual.
      fl_solver_euler_compute_residual(euler, &euler->flow_1, &euler->residual, 0);
      
      // NOTE(cmat): Compute residual norm.
      V3_F64 residual_norm = fl_solver_euler_compute_state_norm2(euler, &euler->residual, range1_u64(0, euler->mesh->cells.len));

      if (lane_index() == 0) {
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

        log_info("TIME %.2g | TIMESTEP %.2g | CFL %.2g | ITERATION %'llu | RESIDUAL %.2g, %.2g, %.2g", time, time_step, CFL, iteration, residual_norm.x, residual_norm.y, residual_norm.z);
      }
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

