// ============================================================================
// fl_scalar_transport.c
//
// Passive scalar transport (advection + anisotropic diffusion) for RedSIM,
// carried by a COMPRESSIBLE velocity field.
//
//   d(phi)/dt + u . grad(phi) = div(D . grad(phi)),    D = diag(Dx, Dy, Dz)
//
// NOTE ON COMPRESSIBILITY (read this before touching the residual kernel):
// The finite-volume discretization naturally computes div(u*phi) (a
// conservative flux divergence through the cell's faces), not u.grad(phi).
// These two only agree when div(u) = 0 (incompressible flow):
//
//   div(u*phi) = phi*div(u) + u.grad(phi)
//
// Since this solver's velocity field comes from a compressible Euler/NS
// solver, div(u) is generally NOT zero -- e.g. strongly nonzero at
// separation/acceleration regions like a bluff body's roof. Left uncorrected,
// a spatially uniform phi under a uniform boundary value is NOT a steady
// state: it silently integrates the local velocity divergence into phi
// every step (phi grows/shrinks wherever the flow is compressing/expanding),
// even with zero diffusivity and a fully-converged flow field.
//
// The fix: accumulate the discrete face-velocity divergence per cell
// alongside the advective flux, and subtract phi*div(u) back out of the
// residual so what's actually solved is d(phi)/dt + u.grad(phi) = ...,
// matching the equation at the top of this file. See
// fl_solver_scalar_compute_residual_range for the implementation --
// search for "velocity_divergence_sum".
//
// This is a structural mirror of FL_Solver_Euler: same ghost/halo pipeline,
// same least-squares gradient + Venkatakrishnan limiter, same SSPRK(4,3)
// explicit stepper, and the same matrix-free JFNK + GMRES(m) + BDF2 implicit
// stepper -- just sized for a single component instead of five, plus an
// anisotropic diffusive flux term the NS equations don't have.
//
// DEPENDENCIES (assumed already available in the translation unit this file
// is included into):
//   - core types/macros: F32, F64, U32, U64, U08, I32, B32, V3F, function,
//     force_inline, Iter_Index, Iter_Range, Range1_U64, range1_u64,
//     range1_u64_len, Zero_Fill, arena_push_count, Arena
//   - lane_index, lane_count, lane_range, lane_barrier, lane_broadcast_ptr,
//     lane_broadcast_type
//   - ipc_rank_index, ipc_rank_barrier, ipc_rank_sum_f64,
//     ipc_rank_minimum_f32, ipc_rank_minimum_f64, ipc_rank_request_list_init,
//     ipc_rank_record_send, ipc_rank_record_receive, IPC_Request_List,
//     IPC_Request_Scope
//   - f32_min/max/sqrt/abs/clamp/div_safe, f32_limit_max, f32_limit_epsilon,
//     f64_min, f64_limit_max
//   - v3f, v3f_add, v3f_sub, v3f_mul, v3f_dot, v3f_len
//   - profiler_begin_function/end_function, log_info, log_zone_start/end,
//     If_Unlikely
//   - UG_Mesh, UG_Cell_Faces, UG_Cell_Gradient, FL_Scale
//   - fl_flux_grad_correct(...)  -- already defined alongside
//     fl_flux_viscous_smagorinsky_LES / fl_flux_viscous_wale_LES. Make sure
//     this file is #included AFTER fluid/fl_build.c in your unity build.
//
// This module never writes velocity -- it only reads whatever three F32*
// arrays you hand it, sized inner+halo+ghost exactly like
// FL_Solver_Euler's primitive_v_x/y/z. The natural choice is to alias those
// arrays directly, so the scalar solver always sees the NS solver's most
// recently computed velocity field.
// ============================================================================

// ------------------------------------------------------------
// #-- Configuration
//
// Scalar transport is far milder than the compressible NS equations: with a
// frozen (externally supplied) velocity field, the only source of
// nonlinearity is the slope limiter. Newton typically converges in 1-3
// iterations. These defaults are looser than the NS tier on purpose --
// tighten if your diffusivities are stiff relative to dt.

#if 0
#define SCALAR_NEWTON_GMRES_M     10     // Krylov restart length
#define SCALAR_NEWTON_GMRES_TOL   1e-2f  // relative -- inexact linear solve is fine
#define SCALAR_NEWTON_MAX_ITERS   5
#define SCALAR_NEWTON_TOL         1e-2f  // relative drop in ||N(phi)|| to accept a Newton step
#define SCALAR_LIMITER_K          5.f
#else

// TODO(cmat): Experimental
#define SCALAR_NEWTON_GMRES_M     10     // Krylov restart length
#define SCALAR_NEWTON_GMRES_TOL   1e-1f  // relative -- inexact linear solve is fine
#define SCALAR_NEWTON_MAX_ITERS   5
#define SCALAR_NEWTON_TOL         1e-1f  // relative drop in ||N(phi)|| to accept a Newton step
#define SCALAR_LIMITER_K          5.f
#endif

// ------------------------------------------------------------
// #-- Boundary conditions

typedef U32 FL_Scalar_Boundary_Type;
enum {
  FL_Scalar_Boundary_Type_Zero_Gradient,  // no-flux / wall: ghost mirrors inner value
  FL_Scalar_Boundary_Type_Dirichlet,      // fixed value, enforced regardless of flow direction
  FL_Scalar_Boundary_Type_Farfield,       // Dirichlet on inflow, zero-gradient on outflow
  FL_Scalar_Boundary_Type_Background_Emission,
};

typedef struct FL_Scalar_Boundary {
  FL_Scalar_Boundary_Type type;
  F32                     dirichlet_value;
  F32                     background_emission_max_height;
} FL_Scalar_Boundary;

typedef struct FL_Scalar_Boundary_Map {
  U64                  map_len;
  FL_Scalar_Boundary  *map_dat;
} FL_Scalar_Boundary_Map;

function void                fl_scalar_boundary_map_init      (FL_Scalar_Boundary_Map *boundary, Arena *arena, U64 len);
function FL_Scalar_Boundary *fl_scalar_boundary_map_by_index  (FL_Scalar_Boundary_Map *boundary, U64 index);

function void fl_scalar_boundary_map_init(FL_Scalar_Boundary_Map *boundary, Arena *arena, U64 len) {
  Zero_Fill(boundary);

  boundary->map_len = len;

  if (lane_index() == 0) {
    boundary->map_dat = arena_push_count(arena, FL_Scalar_Boundary, len);
  }

  lane_broadcast_ptr(&boundary->map_dat, 0);

  // NOTE: default every marker to no-flux, same convention as the Euler
  // boundary map defaulting everything to slip.
  for Iter_Range(it, lane_range(len)) {
    boundary->map_dat[it] = (FL_Scalar_Boundary) { .type = FL_Scalar_Boundary_Type_Zero_Gradient, .dirichlet_value = 0.f };
  }

  lane_barrier();
}

function FL_Scalar_Boundary *fl_scalar_boundary_map_by_index(FL_Scalar_Boundary_Map *boundary, U64 index) {
  FL_Scalar_Boundary *result = 0;
  if (index < boundary->map_len) {
    result = &boundary->map_dat[index];
  }
  return result;
}

// face_normal_velocity is u.n at the boundary face using the mesh's outward
// normal convention (same sign convention fl_boundary_map_ghost relies on):
// negative => inflow, positive => outflow.
force_inline function F32 fl_scalar_boundary_ghost(FL_Scalar_Boundary_Map *bmap, U32 marker_index, F32 phi_inner, F32 face_normal_velocity, V3F ghost_center, FL_Scale *scale) {
  F32 result = phi_inner;
  FL_Scalar_Boundary *boundary = &bmap->map_dat[marker_index];

  switch (boundary->type) {
    case FL_Scalar_Boundary_Type_Zero_Gradient: {
      result = phi_inner;
    } break;

    case FL_Scalar_Boundary_Type_Dirichlet: {
      // NOTE: ghost carries the boundary value directly. This is what gets
      // selected as the upwind state on inflow and what feeds the
      // limiter's neighbor min/max stencil, so it must equal the
      // physical boundary concentration -- the mirrored/doubled value
      // (2*value - phi_inner) massively overshoots whenever phi_inner
      // hasn't caught up to the boundary value yet, which is the common
      // case (e.g. near t=0, or just downstream of an obstacle's wake).
      result = boundary->dirichlet_value;
    } break;

    case FL_Scalar_Boundary_Type_Farfield: {
      if (face_normal_velocity < 0.f) {
        // Inflow: ambient concentration is being carried into the domain.
        result = boundary->dirichlet_value;
      } else {
        // Outflow: let whatever concentration is already inside leave freely.
        result = phi_inner;
      }
    } break;

    case FL_Scalar_Boundary_Type_Background_Emission: {
      F32 z = (ghost_center.z * scale->length) + scale->offset.z;
      if (face_normal_velocity < 0.f) {

        // Outflow: let whatever concentration is already inside leave freely.
        if (z <= boundary->background_emission_max_height) {
          result = boundary->dirichlet_value;
        } else {
          result = 0;
        }
      } else {
        // Outflow: let whatever concentration is already inside leave freely.
        result = phi_inner;
      }
    } break;
  }

  return result;
}

// ------------------------------------------------------------
// #-- State

typedef struct FL_Scalar_Material {
  F32 diffusivity_x;  // Dx
  F32 diffusivity_y;  // Dy
  F32 diffusivity_z;  // Dz
} FL_Scalar_Material;

function void fl_scalar_material_init(FL_Scalar_Material *material, F32 Dx, F32 Dy, F32 Dz) {
  material->diffusivity_x = Dx;
  material->diffusivity_y = Dy;
  material->diffusivity_z = Dz;
}

typedef struct FL_Scalar_State {
  FL_Scalar_Material  material;

  U64  inner_len;
  U64  halo_len;
  U64  ghost_len;

  F32 *phi;
} FL_Scalar_State;

typedef struct FL_Scalar_Gradient_State {
  U64  inner_len;
  U64  halo_len;

  F32 *grad_x;
  F32 *grad_y;
  F32 *grad_z;
} FL_Scalar_Gradient_State;

typedef struct FL_Scalar_Limiter_State {
  U64  inner_len;
  U64  halo_len;

  F32 *limiter;
} FL_Scalar_Limiter_State;

function void fl_scalar_state_init(FL_Scalar_State *fl, FL_Scalar_Material material, UG_Mesh *mesh, B32 store_ghost_halo, Arena *arena) {
  Zero_Fill(fl);

  U64 total_len = mesh->cells.len;
  if (store_ghost_halo) {
    total_len += mesh->halos.len + mesh->ghosts.len;
  }

  F32 *total_dat = 0;
  if (lane_index() == 0) {
    total_dat = arena_push_count(arena, F32, total_len);
  }
  lane_broadcast_ptr(&total_dat, 0);

  fl->material  = material;
  fl->inner_len = mesh->cells.len;
  fl->halo_len  = store_ghost_halo ? mesh->halos.len  : 0;
  fl->ghost_len = store_ghost_halo ? mesh->ghosts.len : 0;
  fl->phi       = total_dat;

  lane_barrier();
}

function void fl_scalar_gradient_state_init(FL_Scalar_Gradient_State *grad, UG_Mesh *mesh, B32 store_halo, Arena *arena) {
  Zero_Fill(grad);

  U64 total_len = mesh->cells.len;
  if (store_halo) {
    total_len += mesh->halos.len;
  }

  grad->inner_len = mesh->cells.len;
  grad->halo_len  = store_halo ? mesh->halos.len : 0;

  F32 *total_dat = 0;
  if (lane_index() == 0) {
    total_dat = arena_push_count(arena, F32, 3 * total_len);
  }
  lane_broadcast_ptr(&total_dat, 0);

  grad->grad_x = total_dat + 0 * total_len;
  grad->grad_y = total_dat + 1 * total_len;
  grad->grad_z = total_dat + 2 * total_len;

  lane_barrier();
}

function void fl_scalar_limiter_state_init(FL_Scalar_Limiter_State *limiter, UG_Mesh *mesh, B32 store_halo, Arena *arena) {
  Zero_Fill(limiter);

  U64 total_len = mesh->cells.len;
  if (store_halo) {
    total_len += mesh->halos.len;
  }

  limiter->inner_len = mesh->cells.len;
  limiter->halo_len  = store_halo ? mesh->halos.len : 0;

  F32 *total_dat = 0;
  if (lane_index() == 0) {
    total_dat = arena_push_count(arena, F32, total_len);
  }
  lane_broadcast_ptr(&total_dat, 0);

  limiter->limiter = total_dat;

  lane_barrier();
}

// Fills inner+halo+ghost with a single value -- the "background concentration"
// initial condition.
function void fl_scalar_state_set_uniform(FL_Scalar_State *state, F32 value) {
  U64 total_len = state->inner_len + state->halo_len + state->ghost_len;
  for Iter_Range(it, lane_range(total_len)) {
    state->phi[it] = value;
  }
  lane_barrier();
}

// ------------------------------------------------------------
// #-- Scale helpers
//
// D has units m^2/s, same as kinematic viscosity nu = mu/rho, so it
// non-dimensionalizes the same way: D_nd = D / (a_ref * L_ref).

function F32 fl_scale_normalize_diffusivity(FL_Scale *scale, F32 diffusivity) {
  F32 result = diffusivity / (scale->sound_speed * scale->length);
  return result;
}

function F32 fl_scale_denormalize_diffusivity(FL_Scale *scale, F32 diffusivity) {
  F32 result = diffusivity * scale->sound_speed * scale->length;
  return result;
}

// A source_mass_rate (kg/s) injected uniformly into a cell of volume
// cell_volume (non-dim) contributes source_mass_rate/cell_volume_dim to
// d(phi)/dt_dim. Converting to non-dim time (t_nd = t_dim * a_ref/L_ref)
// and non-dim volume (V_dim = V_nd * L_ref^3) gives:
//
//   source_nd = source_mass_rate / (V_nd * L_ref^2 * a_ref)
//
function F32 fl_scale_normalize_source_rate(FL_Scale *scale, F32 source_mass_rate, F32 cell_volume) {
  F32 volume_rcp = f32_div_safe(1.f, cell_volume);
  F32 result = source_mass_rate * volume_rcp * f32_div_safe(1.f, scale->length * scale->length * scale->sound_speed);
  return result;
}

// ------------------------------------------------------------
// #-- Solver

typedef struct FL_Solver_Scalar {
  UG_Mesh                 *mesh;
  FL_Scalar_Boundary_Map  *boundary;

  // NOTE: aliases into an externally-owned velocity field (typically your NS
  // solver's flow_1.rho_v1/2/3 and rho), sized inner+halo+ghost. This solver
  // only reads these -- it never writes them, and never advances them in time.
  F32 *rho_velocity_x;
  F32 *rho_velocity_y;
  F32 *rho_velocity_z;
  F32 *rho;

  FL_Scalar_State  phi_1;   // current iterate (RK working state / Newton iterate)
  FL_Scalar_State  phi_2;   // RK stage buffer, or Q^n snapshot for BDF2
  FL_Scalar_State  phi_0;   // Q^{n-1}, needed once BDF2 kicks in
  B32              has_prev_step;

  FL_Scalar_Gradient_State  gradient;
  FL_Scalar_Limiter_State   limiter;
  FL_Scalar_State           residual;

  F64 *cell_spectral_advective_sum;
  F64 *cell_spectral_diffusive_sum;
  F64 *cell_time_step;
  F64 *lane_time_step;

  IPC_Request_List  halo_state_request_list;
  U64               halo_state_receive_len;
  F32              *halo_state_receive_dat;
  U64               halo_state_send_len;
  F32              *halo_state_send_dat;

  IPC_Request_List  halo_gradient_limiter_request_list;
  U64               halo_gradient_limiter_receive_len;
  F32              *halo_gradient_limiter_receive_dat;
  U64               halo_gradient_limiter_send_len;
  F32              *halo_gradient_limiter_send_dat;

  // Matrix-free JFNK + GMRES(m), scalar-sized mirror of FL_Solver_Euler's.
  FL_Scalar_State  newton_state_pert;
  FL_Scalar_State  newton_residual_pert;
  FL_Scalar_State  newton_residual0;
  FL_Scalar_State  newton_rhs;
  FL_Scalar_State  newton_dQ;
  FL_Scalar_State  krylov_basis[SCALAR_NEWTON_GMRES_M + 1];
  FL_Scalar_State  krylov_z;
  F32              hessenberg[(SCALAR_NEWTON_GMRES_M + 1) * SCALAR_NEWTON_GMRES_M];
  F32              givens_cs[SCALAR_NEWTON_GMRES_M];
  F32              givens_sn[SCALAR_NEWTON_GMRES_M];
  F32              gmres_g[SCALAR_NEWTON_GMRES_M + 1];
  F32             *reduce_scratch;


  F32 *source;

  FL_Scale         scale;
} FL_Solver_Scalar;

// ------------------------------------------------------------
// #-- FL_Scalar_State as vector helpers (used by GMRES + the RK stepper)

function void fl_scalar_state_zero(FL_Scalar_State *a, Range1_U64 range) {
  U64 range_len = range1_u64_len(range);
  F32 *a_arr = a->phi;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64 it = range.min + it_range;
    a_arr[it] = 0.f;
  }
  lane_barrier();
}

function void fl_scalar_state_copy(FL_Scalar_State *dst, FL_Scalar_State *src, Range1_U64 range) {
  U64 range_len = range1_u64_len(range);
  F32 *d_arr = dst->phi;
  F32 *s_arr = src->phi;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64 it = range.min + it_range;
    d_arr[it] = s_arr[it];
  }
  lane_barrier();
}

function void fl_scalar_state_copy_scaled(FL_Scalar_State *dst, FL_Scalar_State *src, F32 scale, Range1_U64 range) {
  U64 range_len = range1_u64_len(range);
  F32 *d_arr = dst->phi;
  F32 *s_arr = src->phi;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64 it = range.min + it_range;
    d_arr[it] = s_arr[it] * scale;
  }
  lane_barrier();
}

// dst = alpha*dst + beta*src
function void fl_scalar_state_axpy_in_place(FL_Scalar_State *dst, F32 alpha, F32 beta, FL_Scalar_State *src, Range1_U64 range) {
  U64 range_len = range1_u64_len(range);
  F32 *d_arr = dst->phi;
  F32 *s_arr = src->phi;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64 it = range.min + it_range;
    d_arr[it] = alpha * d_arr[it] + beta * s_arr[it];
  }
  lane_barrier();
}

// dst = clamp(alpha*dst + beta*src, lo, hi), per cell.
//
// Replaces a global-alpha physicality limiter: instead of computing one
// mesh-wide step-length scalar that any single near-floor cell can veto for
// every other cell, this takes the full Newton step everywhere and clips
// only the cells that actually would have gone out of bounds. phi has no
// EOS singularity at 0 (unlike density), so there's no need to protect it
// with a shrinking step -- a hard floor after the update is sufficient and
// doesn't stall propagation elsewhere in the domain.
function void fl_scalar_state_axpy_clamp_in_place(FL_Scalar_State *dst, F32 alpha, F32 beta, FL_Scalar_State *src, F32 lo, F32 hi, Range1_U64 range) {
  U64 range_len = range1_u64_len(range);
  F32 *d_arr = dst->phi;
  F32 *s_arr = src->phi;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64 it = range.min + it_range;
    F32 updated = alpha * d_arr[it] + beta * s_arr[it];
    d_arr[it]   = f32_clamp(updated, lo, hi);
  }
  lane_barrier();
}

function void fl_scalar_state_scale_in_place(FL_Scalar_State *dst, F32 scale, Range1_U64 range) {
  U64 range_len = range1_u64_len(range);
  F32 *d_arr = dst->phi;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64 it = range.min + it_range;
    d_arr[it] *= scale;
  }
  lane_barrier();
}

// ------------------------------------------------------------
// #-- Flux kernels
//
// Split the same way fl_flux_hllc / fl_flux_viscous_wale_LES are split:
// the advective term gets face-reconstructed (MUSCL) left/right values and
// picks upwind; the diffusive term uses cell-CENTER values with a
// directionally-corrected gradient (fl_flux_grad_correct), exactly like the
// viscous flux does for velocity/temperature.

typedef struct FL_Flux_Scalar_Advective {
  F32 flux;        // net upwind advective flux, per unit face area
  F32 lambda_max;  // |u.n|, explicit stability estimate
  F32 vn;          // u.n at the face (signed). Needed to accumulate the
                    // per-cell face-velocity divergence, which corrects for
                    // the compressible dilation term div(u*phi) introduces
                    // but the target equation (u.grad(phi), not div(u*phi))
                    // does not have. See file header comment.
} FL_Flux_Scalar_Advective;

force_inline function FL_Flux_Scalar_Advective fl_flux_scalar_advective(F32 phi_face_L, F32 phi_face_R, V3F velocity_L, V3F velocity_R, V3F normal) {
  FL_Flux_Scalar_Advective result = { };

  V3F velocity_face = v3f_mul(.5f, v3f_add(velocity_L, velocity_R));
  F32 vn            = v3f_dot(velocity_face, normal);
  F32 phi_upwind    = (vn >= 0.f) ? phi_face_L : phi_face_R;

  result.flux       = vn * phi_upwind;
  result.lambda_max = f32_abs(vn);
  result.vn         = vn;
  return result;
}

typedef struct FL_Flux_Scalar_Diffusive {
  F32 flux;              // (D . grad(phi)) . n, per unit face area
  F32 lambda_diffusive;  // stability contribution (already includes area^2/volume)
} FL_Flux_Scalar_Diffusive;

force_inline function FL_Flux_Scalar_Diffusive fl_flux_scalar_diffusive(F32 phi_center_L, F32 phi_center_R, V3F grad_L, V3F grad_R, V3F left_center, V3F right_center, V3F normal, F32 area, F32 left_volume, F32 right_volume, V3F D) {
  FL_Flux_Scalar_Diffusive result = { };

  // If diffusivity is zero (or effectively zero), the flux is exactly zero
  // regardless of geometry -- skip the correction algebra entirely instead
  // of computing it and multiplying by zero. This avoids 0 * Inf -> NaN
  // when dist collapses near degenerate/immersed-boundary geometry (thin
  // walls, sharp corners, coincident ghost centers around buildings).
  F32 D_max = f32_max(f32_max(D.x, D.y), D.z);
  if (D_max <= 0.f) {
    return result;
  }

  V3F center_delta = v3f_sub(right_center, left_center);
  F32 dist         = v3f_len(center_delta);
  F32 dist_rcp     = f32_div_safe(1.f, dist);  // guards degenerate/immersed-boundary geometry
  V3F e_hat        = v3f_mul(dist_rcp, center_delta);

  V3F grad_avg  = v3f_mul(.5f, v3f_add(grad_L, grad_R));
  V3F grad_face = fl_flux_grad_correct(grad_avg, phi_center_L, phi_center_R, e_hat, dist_rcp);

  // Anisotropic Fick's law: q = D . grad(phi), applied component-wise since D is diagonal.
  V3F diffusive_vec = v3f(D.x * grad_face.x, D.y * grad_face.y, D.z * grad_face.z);
  result.flux = v3f_dot(diffusive_vec, normal);

  F32 volume_avg = .5f * (left_volume + right_volume);
  result.lambda_diffusive = D_max * (area * area) / volume_avg;

  return result;
}

// ------------------------------------------------------------
// #-- Halo synchronization (1 component for state, 4 for gradient+limiter)

function void fl_solver_scalar_halo_state_pack_send_data(FL_Solver_Scalar *solver, FL_Scalar_State *state) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;
  for Iter_Range(it_gather, lane_range(mesh->sends.len)) {
    U32 cell_gather = mesh->sends.cell_send[it_gather];
    solver->halo_state_send_dat[it_gather] = state->phi[cell_gather];
  }
  lane_barrier();
  profiler_end_function();
}

function void fl_solver_scalar_halo_state_unpack_receive_data(FL_Solver_Scalar *solver, FL_Scalar_State *state) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;
  for Iter_Range(it_halo, lane_range(mesh->halos.len)) {
    state->phi[mesh->cells.len + it_halo] = solver->halo_state_receive_dat[it_halo];
  }
  lane_barrier();
  profiler_end_function();
}

function void fl_solver_scalar_halo_gradient_limiter_pack_send_data(FL_Solver_Scalar *solver, FL_Scalar_Gradient_State *grad, FL_Scalar_Limiter_State *lim) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;
  for Iter_Range(it_gather, lane_range(mesh->sends.len)) {
    U32 cell_gather = mesh->sends.cell_send[it_gather];
    U64 base = it_gather * 4;
    solver->halo_gradient_limiter_send_dat[base + 0] = grad->grad_x[cell_gather];
    solver->halo_gradient_limiter_send_dat[base + 1] = grad->grad_y[cell_gather];
    solver->halo_gradient_limiter_send_dat[base + 2] = grad->grad_z[cell_gather];
    solver->halo_gradient_limiter_send_dat[base + 3] = lim->limiter[cell_gather];
  }
  lane_barrier();
  profiler_end_function();
}

function void fl_solver_scalar_halo_gradient_limiter_unpack_receive_data(FL_Solver_Scalar *solver, FL_Scalar_Gradient_State *grad, FL_Scalar_Limiter_State *lim) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;
  for Iter_Range(it_halo, lane_range(mesh->halos.len)) {
    U64 base = it_halo * 4;
    grad->grad_x[mesh->cells.len + it_halo] = solver->halo_gradient_limiter_receive_dat[base + 0];
    grad->grad_y[mesh->cells.len + it_halo] = solver->halo_gradient_limiter_receive_dat[base + 1];
    grad->grad_z[mesh->cells.len + it_halo] = solver->halo_gradient_limiter_receive_dat[base + 2];
    lim->limiter[mesh->cells.len + it_halo] = solver->halo_gradient_limiter_receive_dat[base + 3];
  }
  lane_barrier();
  profiler_end_function();
}

function void fl_solver_scalar_halo_state_build_request_list(FL_Solver_Scalar *solver) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;

  ipc_rank_request_list_init(&solver->halo_state_request_list);

  for Iter_Index(it_rank, mesh->halos.block_len) {
    if (it_rank != ipc_rank_index()) {
      Range1_U64 halo_range = mesh->halos.block_range[it_rank];
      U64        halo_len   = range1_u64_len(halo_range);
      if (halo_len) {
        U64 own_tag = ipc_rank_index();
        ipc_rank_record_receive(&solver->halo_state_request_list, halo_len * sizeof(F32), solver->halo_state_receive_dat + halo_range.min, it_rank, own_tag);
      }
    }
  }

  for Iter_Index(it_rank, mesh->sends.block_len) {
    if (it_rank != ipc_rank_index()) {
      Range1_U64 send_range = mesh->sends.block_range[it_rank];
      U64        send_len   = range1_u64_len(send_range);
      if (send_len) {
        U64 other_tag = it_rank;
        ipc_rank_record_send(&solver->halo_state_request_list, send_len * sizeof(F32), solver->halo_state_send_dat + send_range.min, it_rank, other_tag);
      }
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_scalar_halo_gradient_limiter_build_request_list(FL_Solver_Scalar *solver) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;

  ipc_rank_request_list_init(&solver->halo_gradient_limiter_request_list);

  for Iter_Index(it_rank, mesh->halos.block_len) {
    if (it_rank != ipc_rank_index()) {
      Range1_U64 halo_range = mesh->halos.block_range[it_rank];
      U64        halo_len   = range1_u64_len(halo_range);
      if (halo_len) {
        U64 own_tag = ipc_rank_index();
        ipc_rank_record_receive(&solver->halo_gradient_limiter_request_list, halo_len * sizeof(F32) * 4, solver->halo_gradient_limiter_receive_dat + halo_range.min * 4, it_rank, own_tag);
      }
    }
  }

  for Iter_Index(it_rank, mesh->sends.block_len) {
    if (it_rank != ipc_rank_index()) {
      Range1_U64 send_range = mesh->sends.block_range[it_rank];
      U64        send_len   = range1_u64_len(send_range);
      if (send_len) {
        U64 other_tag = it_rank;
        ipc_rank_record_send(&solver->halo_gradient_limiter_request_list, send_len * sizeof(F32) * 4, solver->halo_gradient_limiter_send_dat + send_range.min * 4, it_rank, other_tag);
      }
    }
  }

  lane_barrier();
  profiler_end_function();
}

// ------------------------------------------------------------
// #-- Init

function void fl_solver_scalar_init_implicit(FL_Solver_Scalar *solver, FL_Scalar_Material material, UG_Mesh *mesh, Arena *arena) {
  fl_scalar_state_init(&solver->newton_state_pert,    material, mesh, 1, arena);  // needs halo+ghost
  fl_scalar_state_init(&solver->newton_residual_pert, material, mesh, 0, arena);
  fl_scalar_state_init(&solver->newton_residual0,     material, mesh, 0, arena);
  fl_scalar_state_init(&solver->newton_rhs,           material, mesh, 0, arena);
  fl_scalar_state_init(&solver->newton_dQ,            material, mesh, 0, arena);
  fl_scalar_state_init(&solver->krylov_z,             material, mesh, 0, arena);

  for Iter_Index(i, SCALAR_NEWTON_GMRES_M + 1) {
    fl_scalar_state_init(&solver->krylov_basis[i], material, mesh, 0, arena);
  }

  if (lane_index() == 0) {
    solver->reduce_scratch = arena_push_count(arena, F32, lane_count());
  }
  lane_broadcast_ptr(&solver->reduce_scratch, 0);
}

function void fl_solver_scalar_source_init(FL_Solver_Scalar *solver, UG_Mesh *mesh, Arena *arena) {
  F32 *source_dat = 0;
  if (lane_index() == 0) {
    source_dat = arena_push_count(arena, F32, mesh->cells.len);
  }
  lane_broadcast_ptr(&source_dat, 0);
  solver->source = source_dat;

  for Iter_Range(it, lane_range(mesh->cells.len)) {
    solver->source[it] = 0.f;
  }
  lane_barrier();
}

// rho_velocity_x/y/z and rho must be sized inner+halo+ghost (same layout as
// FL_Solver_Euler's flow_1 state) and kept alive/updated externally.
function void fl_solver_scalar_init(FL_Solver_Scalar *solver, FL_Scalar_Boundary_Map *boundary, FL_Scalar_Material material, UG_Mesh *mesh, F32 *rho_velocity_x, F32 *rho_velocity_y, F32 *rho_velocity_z, F32 *rho, Arena *arena, FL_Scale scale) {
  Zero_Fill(solver);

  solver->mesh       = mesh;
  solver->boundary   = boundary;
  solver->rho_velocity_x = rho_velocity_x;
  solver->rho_velocity_y = rho_velocity_y;
  solver->rho_velocity_z = rho_velocity_z;
  solver->rho            = rho;
  solver->scale          = scale;

  fl_scalar_state_init(&solver->phi_1,    material, mesh, 1, arena);
  fl_scalar_state_init(&solver->phi_2,    material, mesh, 1, arena);
  fl_scalar_state_init(&solver->phi_0,    material, mesh, 1, arena);
  fl_scalar_state_init(&solver->residual, material, mesh, 0, arena);

  fl_scalar_gradient_state_init (&solver->gradient, mesh, 1, arena);
  fl_scalar_limiter_state_init  (&solver->limiter,  mesh, 1, arena);

  solver->halo_state_receive_len = mesh->halos.len;
  solver->halo_state_send_len    = mesh->sends.len;

  solver->halo_gradient_limiter_receive_len = mesh->halos.len * 4;
  solver->halo_gradient_limiter_send_len    = mesh->sends.len * 4;

  if (lane_index() == 0) {
    solver->cell_spectral_advective_sum = arena_push_count(arena, F64, mesh->cells.len);
    solver->cell_spectral_diffusive_sum = arena_push_count(arena, F64, mesh->cells.len);
    solver->cell_time_step              = arena_push_count(arena, F64, mesh->cells.len);
    solver->lane_time_step              = arena_push_count(arena, F64, lane_count());

    solver->halo_state_receive_dat = arena_push_count(arena, F32, solver->halo_state_receive_len);
    solver->halo_state_send_dat    = arena_push_count(arena, F32, solver->halo_state_send_len);

    solver->halo_gradient_limiter_receive_dat = arena_push_count(arena, F32, solver->halo_gradient_limiter_receive_len);
    solver->halo_gradient_limiter_send_dat    = arena_push_count(arena, F32, solver->halo_gradient_limiter_send_len);
  }

  lane_broadcast_ptr(&solver->cell_spectral_advective_sum, 0);
  lane_broadcast_ptr(&solver->cell_spectral_diffusive_sum, 0);
  lane_broadcast_ptr(&solver->cell_time_step, 0);
  lane_broadcast_ptr(&solver->lane_time_step, 0);
  lane_broadcast_ptr(&solver->halo_state_send_dat, 0);
  lane_broadcast_ptr(&solver->halo_state_receive_dat, 0);
  lane_broadcast_ptr(&solver->halo_gradient_limiter_send_dat, 0);
  lane_broadcast_ptr(&solver->halo_gradient_limiter_receive_dat, 0);

  fl_solver_scalar_halo_state_build_request_list             (solver);
  fl_solver_scalar_halo_gradient_limiter_build_request_list  (solver);

  fl_solver_scalar_init_implicit(solver, material, mesh, arena);


  // NEW: allocate + zero solver->source, sized mesh->cells.len.
  fl_solver_scalar_source_init(solver, mesh, arena);
}

// ------------------------------------------------------------
// #-- Source term


function void fl_solver_scalar_source_set_zero(FL_Solver_Scalar *solver) {
  for Iter_Range(it, lane_range(solver->mesh->cells.len)) {
    solver->source[it] = 0.f;
  }
  lane_barrier();
}

// source_mass_rate_kg_s must be an array of length mesh->cells.len (one
// entry per inner cell, in the same cell ordering as everything else).
// Units: kg/s -- the total mass of the transported quantity injected into
// that cell per second (negative = sink/removal). This does the
// non-dimensionalization once and caches the result in solver->source, so
// you only need to call it again when the emission rates actually change
// (e.g. a time-varying schedule), not on every residual evaluation.
function void fl_solver_scalar_source_set(FL_Solver_Scalar *solver, F32 *source_mass_rate_kg_s) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;

  for Iter_Range(it, lane_range(mesh->cells.len)) {
    F32 cell_volume = mesh->cells.volume[it];
    solver->source[it] = fl_scale_normalize_source_rate(&solver->scale, source_mass_rate_kg_s[it], cell_volume);
  }

  lane_barrier();
  profiler_end_function();
}

// Convenience: seed phi_1 / phi_2 / phi_0 all to the same uniform value.
function void fl_solver_scalar_set_uniform(FL_Solver_Scalar *solver, F32 value) {
  fl_scalar_state_set_uniform(&solver->phi_1, value);
  fl_scalar_state_set_uniform(&solver->phi_2, value);
  fl_scalar_state_set_uniform(&solver->phi_0, value);
}

// ------------------------------------------------------------
// #-- Ghost / gradient / limiter / residual

function void fl_solver_scalar_compute_ghost(FL_Solver_Scalar *solver, FL_Scalar_State *state) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;

  for Iter_Range(it, lane_range(mesh->ghosts.len)) {
    U64 cell_parent_index = mesh->ghosts.parent_cell  [it];
    U08 face_parent_index = mesh->ghosts.parent_face  [it];
    U32 marker_index      = mesh->ghosts.marker_index [it];
    U64 phi_ghost_index   = mesh->cells.len + mesh->halos.len + it;

    UG_Cell_Faces *faces = &mesh->cells.faces[cell_parent_index];
    V3F normal   = v3f(faces->normal_x[face_parent_index], faces->normal_y[face_parent_index], faces->normal_z[face_parent_index]);
    V3F velocity = v3f_mul(f32_div_safe(1.f, solver->rho[cell_parent_index]), v3f(solver->rho_velocity_x[cell_parent_index], solver->rho_velocity_y[cell_parent_index], solver->rho_velocity_z[cell_parent_index]));
    F32 vn       = v3f_dot(velocity, normal);

    V3F ghost_center = mesh->cells.center[phi_ghost_index];

    F32 phi_inner = state->phi[cell_parent_index];
    F32 phi_ghost = fl_scalar_boundary_ghost(solver->boundary, marker_index, phi_inner, vn, ghost_center, &solver->scale);

    state->phi[phi_ghost_index] = phi_ghost;
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_scalar_compute_gradient_range(FL_Solver_Scalar *solver, FL_Scalar_State *state, FL_Scalar_Gradient_State *gradient, Range1_U64 range) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);
  UG_Mesh *mesh = solver->mesh;

  for Iter_Range(it_range, lane_range(range_len)) {
    U64                it_cell  = range.min + it_range;
    UG_Cell_Faces     *faces    = &mesh->cells.faces[it_cell];
    UG_Cell_Gradient  *grad_geo = &mesh->cells.gradients[it_cell];

    V3F b = { };
    F32 self_phi = state->phi[it_cell];

    for Iter_Index(it_face, 4) {
      U32 adjacent = faces->adjacent[it_face];
      F32 adj_phi  = state->phi[adjacent];
      V3F w        = grad_geo->weight_dx[it_face];
      b = v3f_add(b, v3f_mul(adj_phi - self_phi, w));
    }

    gradient->grad_x[it_cell] = grad_geo->inv_A_xx * b.x + grad_geo->inv_A_xy * b.y + grad_geo->inv_A_xz * b.z;
    gradient->grad_y[it_cell] = grad_geo->inv_A_xy * b.x + grad_geo->inv_A_yy * b.y + grad_geo->inv_A_yz * b.z;
    gradient->grad_z[it_cell] = grad_geo->inv_A_xz * b.x + grad_geo->inv_A_yz * b.y + grad_geo->inv_A_zz * b.z;
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_scalar_compute_limiter_venkatakrishnan_range(FL_Solver_Scalar *solver, FL_Scalar_State *state, FL_Scalar_Gradient_State *grad, FL_Scalar_Limiter_State *limiter, F32 K, Range1_U64 range) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);
  UG_Mesh *mesh = solver->mesh;

  for Iter_Range(it_range, lane_range(range_len)) {
    U64            it_cell = range.min + it_range;
    UG_Cell_Faces *faces   = &mesh->cells.faces[it_cell];
    F32            volume  = mesh->cells.volume[it_cell];
    F32            eps2    = (K * K * K) * volume;

    F32 phi     = state->phi[it_cell];
    F32 phi_min = phi;
    F32 phi_max = phi;

    for Iter_Index(it_face, 4) {
      U32 adjacent     = faces->adjacent[it_face];
      F32 phi_adjacent = state->phi[adjacent];
      phi_min = f32_min(phi_min, phi_adjacent);
      phi_max = f32_max(phi_max, phi_adjacent);
    }

    V3F cell_grad = v3f(grad->grad_x[it_cell], grad->grad_y[it_cell], grad->grad_z[it_cell]);
    F32 delta_max = phi_max - phi;
    F32 delta_min = phi_min - phi;
    F32 phi_cell  = 1.f;

    for Iter_Index(it_face, 4) {
      V3F face_center  = v3f(faces->center_x[it_face], faces->center_y[it_face], faces->center_z[it_face]);
      V3F center_delta = v3f_sub(face_center, mesh->cells.center[it_cell]);
      F32 delta_face   = v3f_dot(cell_grad, center_delta);
      F32 phi_face     = 1.f;
      F32 vk_epsilon   = 1e-12f;

      if (delta_face > vk_epsilon || delta_face < -vk_epsilon) {
        F32 delta_bound    = (delta_face > 0.f) ? delta_max : delta_min;
        F32 delta_face_rcp = 1.f / delta_face;
        F32 y              = delta_bound * delta_face_rcp;
        F32 eps2_norm      = eps2 * delta_face_rcp * delta_face_rcp;
        phi_face           = (y * y + 2.f * y + eps2_norm) / (y * y + y + 2.f + eps2_norm);
      }

      phi_cell = f32_min(phi_cell, phi_face);
    }

    limiter->limiter[it_cell] = phi_cell;
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_scalar_compute_residual_range(FL_Solver_Scalar *solver, FL_Scalar_State *state, FL_Scalar_State *residual, FL_Scalar_Gradient_State *grad, FL_Scalar_Limiter_State *limiter, Range1_U64 range, B32 compute_time_step) {
  profiler_begin_function();

  U64 range_len = range1_u64_len(range);
  if (compute_time_step) {
    for Iter_Range(it_range, lane_range(range_len)) {
      U64 it_cell = range.min + it_range;
      solver->cell_spectral_advective_sum[it_cell] = 0.0;
      solver->cell_spectral_diffusive_sum[it_cell] = 0.0;
    }
  }

  UG_Mesh *mesh = solver->mesh;
  V3F D = v3f(state->material.diffusivity_x, state->material.diffusivity_y, state->material.diffusivity_z);

  for Iter_Range(it_range, lane_range(range_len)) {
    U64            it_cell       = range.min + it_range;
    UG_Cell_Faces *faces         = &mesh->cells.faces[it_cell];
    F32            cell_residual = 0.f;

    F32 phi_left     = state->phi[it_cell];
    V3F grad_left    = v3f(grad->grad_x[it_cell], grad->grad_y[it_cell], grad->grad_z[it_cell]);
    F32 limiter_left = limiter->limiter[it_cell];

    V3F cell_center   = mesh->cells.center[it_cell];
    F32 cell_volume   = mesh->cells.volume[it_cell];
    V3F velocity_left = v3f_mul(f32_div_safe(1.f, solver->rho[it_cell]), v3f(solver->rho_velocity_x[it_cell], solver->rho_velocity_y[it_cell], solver->rho_velocity_z[it_cell]));

    // NOTE(compressibility): discrete divergence of the face-averaged
    // velocity field through this cell -- sum(area * vn) over the closed
    // cell boundary. For a truly incompressible field this sums to ~0 and
    // the correction below is a no-op; for a compressible field (this
    // solver) it is generally nonzero, and represents exactly the spurious
    // "phi * div(u)" dilation term that div(u*phi) introduces but the
    // target equation d(phi)/dt + u.grad(phi) = D*lap(phi) does not have.
    // See the file header comment for the derivation.
    F32 velocity_divergence_sum = 0.f;

    for Iter_Index(it_face, 4) {
      U32 adjacent    = faces->adjacent[it_face];
      V3F normal      = v3f(faces->normal_x[it_face], faces->normal_y[it_face], faces->normal_z[it_face]);
      F32 area        = faces->area[it_face];
      V3F face_center = v3f(faces->center_x[it_face], faces->center_y[it_face], faces->center_z[it_face]);

      // Left-side MUSCL reconstruction to the face -- advective upwind only.
      V3F center_delta_l = v3f_sub(face_center, cell_center);
      F32 phi_face_left  = phi_left + limiter_left * v3f_dot(grad_left, center_delta_l);

      B32 is_ghost = adjacent >= (mesh->cells.len + mesh->halos.len);

      F32 phi_right_center;
      F32 phi_face_right;
      V3F grad_right;

      if (!is_ghost) {
        F32 limiter_right  = limiter->limiter[adjacent];
        grad_right          = v3f(grad->grad_x[adjacent], grad->grad_y[adjacent], grad->grad_z[adjacent]);
        V3F center_delta_r  = v3f_sub(face_center, mesh->cells.center[adjacent]);
        phi_right_center    = state->phi[adjacent];
        phi_face_right      = phi_right_center + limiter_right * v3f_dot(grad_right, center_delta_r);
      } else {
        // Ghosts carry no gradient -- first-order fallback, same as the NS solver.
        grad_right       = v3f(0.f, 0.f, 0.f);
        phi_right_center = state->phi[adjacent];
        phi_face_right   = phi_right_center;
      }

      V3F velocity_right = v3f_mul(f32_div_safe(1.f, solver->rho[adjacent]), v3f(solver->rho_velocity_x[adjacent], solver->rho_velocity_y[adjacent], solver->rho_velocity_z[adjacent]));
      V3F right_center   = mesh->cells.center[adjacent];
      F32 right_volume   = mesh->cells.volume[adjacent];

      FL_Flux_Scalar_Advective flux_adv  = fl_flux_scalar_advective(phi_face_left, phi_face_right, velocity_left, velocity_right, normal);
      FL_Flux_Scalar_Diffusive flux_diff = fl_flux_scalar_diffusive(phi_left, phi_right_center, grad_left, grad_right, cell_center, right_center, normal, area, cell_volume, right_volume, D);

      F32 flux_total = flux_adv.flux - flux_diff.flux;
      cell_residual  = cell_residual - area * flux_total;
      velocity_divergence_sum += area * flux_adv.vn;

      if (compute_time_step) {
        solver->cell_spectral_advective_sum[it_cell] += (F64)area * (F64)flux_adv.lambda_max;
        solver->cell_spectral_diffusive_sum[it_cell] += (F64)flux_diff.lambda_diffusive;
      }
    }

    // Cancel the spurious phi*div(u) dilation term picked up by the
    // conservative flux-divergence discretization: div(u*phi) = phi*div(u)
    // + u.grad(phi), and only the second piece belongs in this equation.
    cell_residual = cell_residual + phi_left * velocity_divergence_sum;

    F32 volume_rcp = 1.f / cell_volume;
    residual->phi[it_cell] = cell_residual * volume_rcp + solver->source[it_cell];

    if (compute_time_step) {
      solver->cell_time_step[it_cell] = (F64)cell_volume / (solver->cell_spectral_advective_sum[it_cell] + solver->cell_spectral_diffusive_sum[it_cell]);
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_scalar_compute_residual(FL_Solver_Scalar *solver, FL_Scalar_State *state, FL_Scalar_State *residual, B32 compute_time_step) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;

  lane_barrier();
  fl_solver_scalar_halo_state_pack_send_data(solver, state);

  IPC_Request_Scope(&solver->halo_state_request_list) {
    fl_solver_scalar_compute_ghost(solver, state);

    fl_solver_scalar_compute_gradient_range               (solver, state, &solver->gradient, mesh->groups.cells_interior);
    fl_solver_scalar_compute_limiter_venkatakrishnan_range (solver, state, &solver->gradient, &solver->limiter, SCALAR_LIMITER_K, mesh->groups.cells_interior);
  }

  fl_solver_scalar_halo_state_unpack_receive_data(solver, state);

  fl_solver_scalar_compute_gradient_range               (solver, state, &solver->gradient, mesh->groups.cells_boundary);
  fl_solver_scalar_compute_limiter_venkatakrishnan_range (solver, state, &solver->gradient, &solver->limiter, SCALAR_LIMITER_K, mesh->groups.cells_boundary);

  fl_solver_scalar_halo_gradient_limiter_pack_send_data(solver, &solver->gradient, &solver->limiter);

  IPC_Request_Scope(&solver->halo_gradient_limiter_request_list) {
    fl_solver_scalar_compute_residual_range(solver, state, residual, &solver->gradient, &solver->limiter, mesh->groups.cells_interior, compute_time_step);
  }

  fl_solver_scalar_halo_gradient_limiter_unpack_receive_data(solver, &solver->gradient, &solver->limiter);

  fl_solver_scalar_compute_residual_range(solver, state, residual, &solver->gradient, &solver->limiter, mesh->groups.cells_boundary, compute_time_step);

  lane_barrier();
  profiler_end_function();
}

// ------------------------------------------------------------
// #-- Global reductions

function F32 fl_solver_scalar_global_dot(FL_Solver_Scalar *solver, FL_Scalar_State *a, FL_Scalar_State *b, Range1_U64 range) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);
  F32 *a_arr = a->phi;
  F32 *b_arr = b->phi;

  F32 local_dot = 0.f;
  for Iter_Range(it_range, lane_range(range_len)) {
    U64 it = range.min + it_range;
    local_dot += a_arr[it] * b_arr[it];
  }

  solver->reduce_scratch[lane_index()] = local_dot;
  lane_barrier();

  F32 global_dot = 0.f;
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      global_dot += solver->reduce_scratch[it];
    }
  }
  lane_broadcast_type(&global_dot, 0);
  global_dot = (F32)ipc_rank_sum_f64((F64)global_dot);

  lane_barrier();
  profiler_end_function();
  return global_dot;
}

function F32 fl_solver_scalar_global_norm(FL_Solver_Scalar *solver, FL_Scalar_State *a, Range1_U64 range) {
  return f32_sqrt(fl_solver_scalar_global_dot(solver, a, a, range));
}

function F32 fl_solver_scalar_global_min(FL_Solver_Scalar *solver, F32 lane_local_min) {
  profiler_begin_function();

  solver->reduce_scratch[lane_index()] = lane_local_min;
  lane_barrier();

  F32 global_min = f32_limit_max;
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      global_min = f32_min(global_min, solver->reduce_scratch[it]);
    }
  }
  lane_broadcast_type(&global_min, 0);
  global_min = ipc_rank_minimum_f32(global_min);

  lane_barrier();
  profiler_end_function();
  return global_min;
}

function F64 fl_solver_scalar_compute_global_time_step(FL_Solver_Scalar *solver, F64 *time_steps) {
  profiler_begin_function();

  F64 lane_time_step = f64_limit_max;
  for Iter_Range(it, lane_range(solver->mesh->cells.len)) {
    lane_time_step = f64_min(lane_time_step, time_steps[it]);
  }
  solver->lane_time_step[lane_index()] = lane_time_step;
  lane_barrier();

  F64 global_time_step = f64_limit_max;
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      global_time_step = f64_min(global_time_step, solver->lane_time_step[it]);
    }
  }
  lane_broadcast_type(&global_time_step, 0);
  global_time_step = ipc_rank_minimum_f64(global_time_step);

  profiler_end_function();
  return global_time_step;
}

// ------------------------------------------------------------
// #-- Explicit: SSP-RK(4,3)
// Same scheme as fl_solver_euler_solve_global_step_SSP_RK_4_3, phi instead of Q.

function void fl_solver_scalar_global_step(FL_Solver_Scalar *solver, FL_Scalar_State *state_dst, FL_Scalar_State *state_src, FL_Scalar_State *residual, F32 time_scale, F64 time_step) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;
  F64 scaled_time_step = time_scale * time_step;

  F32 *src = state_src->phi;
  F32 *dst = state_dst->phi;
  F32 *res = residual->phi;

  for Iter_Range(it, lane_range(mesh->cells.len)) {
    dst[it] = (F32)(src[it] + scaled_time_step * res[it]);
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_scalar_global_step_2(FL_Solver_Scalar *solver, FL_Scalar_State *state_dst, F32 state_1_coeff, FL_Scalar_State *state_1, F32 state_2_coeff, FL_Scalar_State *state_2, FL_Scalar_State *residual, F32 time_scale, F64 time_step) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;
  F64 scaled_time_step = time_scale * time_step;

  F32 *s1  = state_1->phi;
  F32 *s2  = state_2->phi;
  F32 *dst = state_dst->phi;
  F32 *res = residual->phi;

  for Iter_Range(it, lane_range(mesh->cells.len)) {
    dst[it] = (F32)(state_1_coeff * s1[it] + state_2_coeff * s2[it] + scaled_time_step * res[it]);
  }

  lane_barrier();
  profiler_end_function();
}

function F64 fl_solver_scalar_solve_global_step_SSP_RK_4_3(FL_Solver_Scalar *solver, F32 CFL, F32 max_time_step) {
  profiler_begin_function();

  CFL *= 2.f;

  // R(phi_1)
  fl_solver_scalar_compute_residual(solver, &solver->phi_1, &solver->residual, 1);

  F64 time_step = fl_solver_scalar_compute_global_time_step(solver, solver->cell_time_step);
  time_step *= CFL;
  time_step  = f32_min(time_step, max_time_step);

  // phi_2 = phi_1 + dt/2 * R(phi_1)
  fl_solver_scalar_global_step(solver, &solver->phi_2, &solver->phi_1, &solver->residual, .5f, time_step);
  fl_solver_scalar_compute_residual(solver, &solver->phi_2, &solver->residual, 0);

  // phi_2 = phi_2 + dt/2 * R(phi_2)
  fl_solver_scalar_global_step(solver, &solver->phi_2, &solver->phi_2, &solver->residual, .5f, time_step);
  fl_solver_scalar_compute_residual(solver, &solver->phi_2, &solver->residual, 0);

  // phi_2 = 2/3*phi_1 + 1/3*[phi_2 + dt/2*R(phi_2)]
  fl_solver_scalar_global_step_2(solver, &solver->phi_2, 2.f/3.f, &solver->phi_1, 1.f/3.f, &solver->phi_2, &solver->residual, 1.f/6.f, time_step);
  fl_solver_scalar_compute_residual(solver, &solver->phi_2, &solver->residual, 0);

  // phi_1 = phi_2 + dt/2 * R(phi_2)
  fl_solver_scalar_global_step(solver, &solver->phi_1, &solver->phi_2, &solver->residual, .5f, time_step);

  profiler_end_function();
  return time_step;
}

function F32 fl_solver_scalar_solve(FL_Solver_Scalar *solver, F32 time_target) {
  profiler_begin_function();
  log_zone_start("Solving scalar transport (explicit SSP-RK(4,3))");

  ipc_rank_barrier();
  F32 CFL = 0.85f;

  F64 time      = 0;
  U64 iteration = 0;

  static B32 residual_norm_init  = 0;
  static F32 residual_norm_first = 0.f;

  while (time < time_target) {
    F64 max_time_step = time_target - time;
    F64 time_step      = fl_solver_scalar_solve_global_step_SSP_RK_4_3(solver, CFL, (F32)max_time_step);
    time      += time_step;
    iteration += 1;

    fl_solver_scalar_compute_residual(solver, &solver->phi_1, &solver->residual, 0);
    F32 residual_dot  = fl_solver_scalar_global_dot(solver, &solver->residual, &solver->residual, range1_u64(0, solver->mesh->cells.len));
    F32 residual_norm = f32_sqrt(residual_dot / (F32)solver->mesh->cells.len);

    if (lane_index() == 0) {
      If_Unlikely (!residual_norm_init) {
        residual_norm_init  = 1;
        residual_norm_first = residual_norm;
      }
      F32 residual_norm_rel = residual_norm / f32_max(residual_norm_first, 1e-30f);
      log_info("SCALAR TIME %.2g | TIMESTEP %.2g | ITERATION %'llu | RESIDUAL %.2g", time, time_step, iteration, residual_norm_rel);
    }
  }

  lane_barrier();
  log_zone_end();
  profiler_end_function();
  return (F32)time;
}

// ------------------------------------------------------------
// #-- Implicit: matrix-free JFNK + GMRES(m), same structure as the NS solver

function void fl_solver_scalar_compute_residual_frozen(FL_Solver_Scalar *solver, FL_Scalar_State *state, FL_Scalar_State *residual) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;

  lane_barrier();
  fl_solver_scalar_halo_state_pack_send_data(solver, state);

  IPC_Request_Scope(&solver->halo_state_request_list) {
    fl_solver_scalar_compute_ghost(solver, state);
  }

  fl_solver_scalar_halo_state_unpack_receive_data(solver, state);

  // NOTE: solver->gradient / solver->limiter intentionally NOT recomputed here --
  // frozen at the current Newton iterate, same "frozen Jacobian-vector product"
  // approximation as fl_solver_euler_compute_residual_frozen. The compressibility
  // correction inside fl_solver_scalar_compute_residual_range is linear in
  // phi_left, so it differentiates cleanly through the JFNK finite difference.
  fl_solver_scalar_compute_residual_range(solver, state, residual, &solver->gradient, &solver->limiter, mesh->groups.cells_interior, 0);
  fl_solver_scalar_compute_residual_range(solver, state, residual, &solver->gradient, &solver->limiter, mesh->groups.cells_boundary, 0);

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_scalar_jacobian_vector_product_global(FL_Solver_Scalar *solver, FL_Scalar_State *Q0, FL_Scalar_State *R0, FL_Scalar_State *v, FL_Scalar_State *Jv_out) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;
  Range1_U64 range = range1_u64(0, mesh->cells.len);

  F32 v_norm = fl_solver_scalar_global_norm(solver, v, range);
  F32 q_norm = fl_solver_scalar_global_norm(solver, Q0, range);
  F32 eps    = f32_sqrt(f32_limit_epsilon) * (1.f + q_norm) / f32_max(v_norm, 1e-30f);

  F32 *q0 = Q0->phi;
  F32 *vv = v->phi;
  F32 *qp = solver->newton_state_pert.phi;
  for Iter_Range(it, lane_range(mesh->cells.len)) {
    qp[it] = q0[it] + eps * vv[it];
  }
  lane_barrier();

  fl_solver_scalar_compute_residual_frozen(solver, &solver->newton_state_pert, &solver->newton_residual_pert);

  F32 *r0 = R0->phi;
  F32 *rp = solver->newton_residual_pert.phi;
  F32 *jv = Jv_out->phi;
  for Iter_Range(it, lane_range(mesh->cells.len)) {
    jv[it] = (rp[it] - r0[it]) / eps;
  }
  lane_barrier();
  profiler_end_function();
}

// M^-1 r ~= r / (1/dt + spectral_radius_sum / volume), single global dt.
function void fl_solver_scalar_apply_preconditioner_global(FL_Solver_Scalar *solver, FL_Scalar_State *r, FL_Scalar_State *z, F32 time_coeff) {
  UG_Mesh *mesh = solver->mesh;
  F32 *r_arr = r->phi;
  F32 *z_arr = z->phi;
  for Iter_Range(it, lane_range(mesh->cells.len)) {
    F32 volume   = mesh->cells.volume[it];
    F32 spectral = solver->cell_spectral_advective_sum[it] + solver->cell_spectral_diffusive_sum[it];
    F32 diag     = time_coeff + spectral / volume;
    z_arr[it]    = r_arr[it] / diag;
  }
  lane_barrier();
}

// H is row-major with row stride SCALAR_NEWTON_GMRES_M. Same Givens-rotation
// update as gmres_apply_givens_and_update, just addressed with the scalar
// stride so it can coexist with the NS solver's GMRES in the same TU.
function F32 fl_scalar_gmres_apply_givens_and_update(F32 *H, F32 *cs, F32 *sn, F32 *g, U32 j) {
  for Iter_Index(i, j) {
    F32 h_i  = H[i * SCALAR_NEWTON_GMRES_M + j];
    F32 h_i1 = H[(i + 1) * SCALAR_NEWTON_GMRES_M + j];
    H[i * SCALAR_NEWTON_GMRES_M + j]       =  cs[i] * h_i + sn[i] * h_i1;
    H[(i + 1) * SCALAR_NEWTON_GMRES_M + j] = -sn[i] * h_i + cs[i] * h_i1;
  }

  F32 h_jj  = H[j * SCALAR_NEWTON_GMRES_M + j];
  F32 h_j1j = H[(j + 1) * SCALAR_NEWTON_GMRES_M + j];
  F32 denom = f32_sqrt(h_jj * h_jj + h_j1j * h_j1j);

  F32 c = 1.f, s = 0.f;
  if (denom > 1e-30f) {
    c = h_jj  / denom;
    s = h_j1j / denom;
  }
  cs[j] = c;
  sn[j] = s;

  H[j * SCALAR_NEWTON_GMRES_M + j]       = c * h_jj + s * h_j1j;
  H[(j + 1) * SCALAR_NEWTON_GMRES_M + j] = 0.f;

  F32 g_j  = g[j];
  F32 g_j1 = g[j + 1];
  g[j]     =  c * g_j + s * g_j1;
  g[j + 1] = -s * g_j + c * g_j1;

  return f32_abs(g[j + 1]);
}

function void fl_scalar_gmres_back_substitute(F32 *H, F32 *g, U32 m_used, F32 *y) {
  for (I32 i = (I32)m_used - 1; i >= 0; i -= 1) {
    F32 sum = g[i];
    for (U32 k = i + 1; k < m_used; k += 1) {
      sum -= H[i * SCALAR_NEWTON_GMRES_M + k] * y[k];
    }
    y[i] = sum / H[i * SCALAR_NEWTON_GMRES_M + i];
  }
}

// Solves A*dphi = b where A(v) = v/dt - Jv(v), right-preconditioned by M^-1.
function F32 fl_solver_scalar_gmres_solve_global(FL_Solver_Scalar *solver, FL_Scalar_State *Q0, FL_Scalar_State *R0, FL_Scalar_State *b, F32 time_coeff, F32 gmres_tol, FL_Scalar_State *dQ) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;
  Range1_U64 range = range1_u64(0, mesh->cells.len);

  F32 beta = fl_solver_scalar_global_norm(solver, b, range);
  if (beta < 1e-30f) {
    fl_scalar_state_zero(dQ, range);
    profiler_end_function();
    return 0.f;
  }

  fl_scalar_state_copy_scaled(&solver->krylov_basis[0], b, 1.f / beta, range);
  for Iter_Index(it, SCALAR_NEWTON_GMRES_M + 1) {
    solver->gmres_g[it] = 0;
  }
  solver->gmres_g[0] = beta;

  U32 m_used = SCALAR_NEWTON_GMRES_M;
  for Iter_Index(j, SCALAR_NEWTON_GMRES_M) {

    fl_solver_scalar_apply_preconditioner_global(solver, &solver->krylov_basis[j], &solver->krylov_z, time_coeff);
    fl_solver_scalar_jacobian_vector_product_global(solver, Q0, R0, &solver->krylov_z, &solver->krylov_basis[j + 1]);
    fl_scalar_state_axpy_in_place(&solver->krylov_basis[j + 1], -1.f, time_coeff, &solver->krylov_z, range);

    for Iter_Index(i, j + 1) {
      F32 h_ij = fl_solver_scalar_global_dot(solver, &solver->krylov_basis[j + 1], &solver->krylov_basis[i], range);
      solver->hessenberg[i * SCALAR_NEWTON_GMRES_M + j] = h_ij;
      fl_scalar_state_axpy_in_place(&solver->krylov_basis[j + 1], 1.f, -h_ij, &solver->krylov_basis[i], range);
    }

    F32 h_next = fl_solver_scalar_global_norm(solver, &solver->krylov_basis[j + 1], range);
    solver->hessenberg[(j + 1) * SCALAR_NEWTON_GMRES_M + j] = h_next;

    if (h_next > 1e-30f) {
      fl_scalar_state_scale_in_place(&solver->krylov_basis[j + 1], 1.f / h_next, range);
    }

    F32 residual_estimate = fl_scalar_gmres_apply_givens_and_update(solver->hessenberg, solver->givens_cs, solver->givens_sn, solver->gmres_g, j);

    if (residual_estimate / beta < gmres_tol || h_next < 1e-30f) {
      m_used = j + 1;
      break;
    }
  }

  F32 y[SCALAR_NEWTON_GMRES_M];
  fl_scalar_gmres_back_substitute(solver->hessenberg, solver->gmres_g, m_used, y);

  fl_scalar_state_zero(dQ, range);
  for Iter_Index(i, m_used) {
    fl_solver_scalar_apply_preconditioner_global(solver, &solver->krylov_basis[i], &solver->krylov_z, time_coeff);
    fl_scalar_state_axpy_in_place(dQ, 1.f, y[i], &solver->krylov_z, range);
  }

  F32 relative_residual = f32_abs(solver->gmres_g[m_used] / beta);

  if (lane_index() == 0) {
    log_info("    SCALAR GMRES: %u/%u iters | rel_residual %.3g", m_used, SCALAR_NEWTON_GMRES_M, relative_residual);
  }

  profiler_end_function();
  return relative_residual;
}

function F32 fl_solver_scalar_solve_global_step_backward_euler_BDF2_JFNK(FL_Solver_Scalar *solver, F32 CFL, F32 max_dt, B32 *out_converged) {
  profiler_begin_function();
  UG_Mesh *mesh = solver->mesh;
  Range1_U64 range = range1_u64(0, mesh->cells.len);

  fl_solver_scalar_compute_residual(solver, &solver->phi_1, &solver->residual, 1);
  F32 dt = fl_solver_scalar_compute_global_time_step(solver, solver->cell_time_step) * CFL;
  dt = f32_min(dt, max_dt);

  // BDF2 once we have two prior levels; plain backward Euler to bootstrap the
  // first step, since BDF2 needs phi^{n-1}.
  F32 time_coeff = solver->has_prev_step ? f32_div_safe(1.5f, dt) : f32_div_safe(1.f, dt);

  if (solver->has_prev_step) {
    fl_scalar_state_copy(&solver->phi_0, &solver->phi_2, range);
  }
  fl_scalar_state_copy(&solver->phi_2, &solver->phi_1, range);

  F32 n0_norm        = 0.f;
  F32 n_norm         = 0.f;
  F32 prev_ratio     = 1.f;
  U32 stagnant_count = 0;
  U32 k_used         = SCALAR_NEWTON_MAX_ITERS;
  B32 stagnated      = 0;

  for Iter_Index(k, SCALAR_NEWTON_MAX_ITERS) {

    fl_solver_scalar_compute_residual(solver, &solver->phi_1, &solver->newton_residual0, 1);

    if (solver->has_prev_step) {
      F32 *r_arr    = solver->newton_residual0.phi;
      F32 *q_arr    = solver->phi_1.phi;
      F32 *qn_arr   = solver->phi_2.phi;
      F32 *qnm1_arr = solver->phi_0.phi;
      F32 *b_arr    = solver->newton_rhs.phi;
      for Iter_Range(it, lane_range(mesh->cells.len)) {
        b_arr[it] = r_arr[it] - f32_div_safe((1.5f * q_arr[it] - 2.f * qn_arr[it] + 0.5f * qnm1_arr[it]), dt);
      }
    } else {
      F32 *r_arr  = solver->newton_residual0.phi;
      F32 *q_arr  = solver->phi_1.phi;
      F32 *qn_arr = solver->phi_2.phi;
      F32 *b_arr  = solver->newton_rhs.phi;
      for Iter_Range(it, lane_range(mesh->cells.len)) {
        b_arr[it] = r_arr[it] - f32_div_safe(q_arr[it] - qn_arr[it], dt);
      }
    }
    lane_barrier();

    n_norm = fl_solver_scalar_global_norm(solver, &solver->newton_rhs, range);
    if (k == 0) {
      n0_norm = n_norm;
    }

    F32 ratio = n_norm / f32_max(n0_norm, 1e-30f);
    if (ratio < SCALAR_NEWTON_TOL) {
      k_used = k;
      break;
    }

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

    F32 gmres_tol = f32_clamp(0.1f * ratio, SCALAR_NEWTON_TOL, 1e-1f);

    if (lane_index() == 0) {
      log_info("    SCALAR Newton k=%u | forcing gmres_tol=%.2g", k, gmres_tol);
    }

    fl_solver_scalar_gmres_solve_global(solver, &solver->phi_1, &solver->newton_residual0, &solver->newton_rhs, time_coeff, gmres_tol, &solver->newton_dQ);

    // Take the full Newton step, clamping only the cells that would go
    // negative. If phi is a bounded fraction in your setup (e.g. [0,1]),
    // change f32_limit_max below to 1.f to also clip overshoot.
    fl_scalar_state_axpy_clamp_in_place(&solver->phi_1, 1.f, 1.f, &solver->newton_dQ, 0.f, f32_limit_max, range);
  }

  if (lane_index() == 0) {
    B32 converged = (k_used < SCALAR_NEWTON_MAX_ITERS) && !stagnated;
    const char *status = converged ? "" : (stagnated ? "  *** STAGNATED ***" : "  *** DID NOT CONVERGE ***");
    log_info("  SCALAR Newton: %u/%u iters | residual ratio %.3g%s", k_used, SCALAR_NEWTON_MAX_ITERS, n_norm / f32_max(n0_norm, 1e-30f), status);
    *out_converged = converged;
  }

  lane_broadcast_type(out_converged, 0);

  solver->has_prev_step = 1;

  profiler_end_function();
  return dt;
}

function F32 fl_solver_scalar_solve_implicit(FL_Solver_Scalar *solver, F32 time_target) {
  profiler_begin_function();
  log_zone_start("Solving scalar transport (implicit BDF2/JFNK)");

  ipc_rank_barrier();

  F32 CFL_max    = 10.0f;
  F32 CFL_growth = 1.03f;
  F32 CFL_min    = 0.05f;
  static F32 CFL = 0.1f;

  F64 time      = 0;
  U64 iteration = 0;

  while (time < time_target) {
    F32 max_time_step = time_target - time;
    // F32 time_step      = fl_solver_scalar_solve_global_step_backward_euler_BDF2_JFNK(solver, CFL, max_time_step);
    B32 step_converged = 0;

    F32 time_step      = fl_solver_scalar_solve_global_step_backward_euler_BDF2_JFNK(solver, CFL, max_time_step, &step_converged);

    time      += time_step;
    iteration += 1;

    if (lane_index() == 0) {
      CFL = f32_min(CFL_max, CFL * CFL_growth);
      // CFL = step_converged ? f32_min(CFL_max, CFL * CFL_growth) : f32_max(CFL_min, CFL * 0.5f);
    }

    lane_broadcast_type(&CFL, 0);

    fl_solver_scalar_compute_residual(solver, &solver->phi_1, &solver->residual, 0);
    F32 residual_dot  = fl_solver_scalar_global_dot(solver, &solver->residual, &solver->residual, range1_u64(0, solver->mesh->cells.len));
    F32 residual_norm = f32_sqrt(residual_dot / (F32)solver->mesh->cells.len);

    if (lane_index() == 0) {
      log_info("SCALAR TIME %.2g | TIMESTEP %.2g | CFL %.2g | ITERATION %'llu | RESIDUAL %.2g", time, time_step, CFL, iteration, residual_norm);
    }
  }

  lane_barrier();
  log_zone_end();
  profiler_end_function();
  return (F32)time;
}
