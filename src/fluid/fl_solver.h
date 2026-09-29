#if 0
#define NEWTON_GMRES_M       30     // Krylov restart length
#define NEWTON_GMRES_TOL     1e-2f  // relative — Newton only needs an inexact linear solve
#define NEWTON_MAX_ITERS     10
#define NEWTON_TOL           1e-2f  // relative drop in ||N(Q)|| to accept a Newton step

#elif 0 // NOTE(cmat): Works!
#define NEWTON_GMRES_M       15     // Krylov restart length
#define NEWTON_GMRES_TOL     1e-2f  // relative — Newton only needs an inexact linear solve
#define NEWTON_MAX_ITERS     5
#define NEWTON_TOL           1e-2f  // relative drop in ||N(Q)|| to accept a Newton step
#else
#define NEWTON_GMRES_M       40     // Krylov restart length
#define NEWTON_GMRES_TOL     1e-2f  // relative — Newton only needs an inexact linear solve
#define NEWTON_MAX_ITERS     15
#define NEWTON_TOL           1e-2f  // relative drop in ||N(Q)|| to accept a Newton step
#endif


// NOTE(cmat): LES SGS model used by BOTH the viscous flux (fl_solver_euler_compute_residual_range) and the
// - per-cell eddy viscosity handed to the scalar solver (fl_solver_euler_compute_eddy_viscosity).
// - 1 = Smagorinsky, 0 = WALE.
#ifndef FL_LES_USE_SMAGORINSKY
#define FL_LES_USE_SMAGORINSKY 1
#endif

typedef struct FL_Solver_Euler {
  UG_Mesh            *mesh;
  FL_Boundary_Map    *boundary;
  FL_Scale            scale;

  FL_State            flow_1;
  FL_State            flow_2;

  FL_Gradient_State   gradient;
  FL_Limiter_State    limiter;
  FL_State            residual;

  // NOTE(cmat): Primitive variable storage.
  // - These come up everywhere in computations, 
  // - so we compute them in a single pass then refer to them.
  // - These are stored for halos and ghosts.
  F32                *primitive_v_x;
  F32                *primitive_v_y;
  F32                *primitive_v_z;
  F32                *primitive_pressure;

  // NOTE(cmat): Time-Step for each cell.
  F64                *cell_spectral_inviscid_sum;
  F64                *cell_spectral_viscous_sum;
  F64                *cell_time_step;
  F64                *lane_time_step;
  V3_F64             *lane_state_norm2;

  // NOTE(cmat): Halo state synchronization.
  IPC_Request_List    halo_state_request_list;
  U64                 halo_state_receive_len;
  F32                *halo_state_receive_dat;
  U64                 halo_state_send_len;
  F32                *halo_state_send_dat;

  // NOTE(cmat): Halo gradient + limiter synchronization.
  IPC_Request_List    halo_gradient_limiter_request_list;
  U64                 halo_gradient_limiter_receive_len;
  F32                *halo_gradient_limiter_receive_dat;
  U64                 halo_gradient_limiter_send_len;
  F32                *halo_gradient_limiter_send_dat;

  // NOTE(cmat): Per-cell SGS kinematic eddy viscosity nu_t = mu_sgs / rho (NON-dimensional, flow solver
  // - units), sized inner+halo+ghost (ghost entries stay 0). Filled by fl_solver_euler_compute_eddy_viscosity
  // - from the current velocity gradients using the same model as the viscous flux. Read by the scalar solver.
  F32                *eddy_viscosity;

  // TODO(cmat): Temporary.
  V3F                 gravity;

  // TODO(cmat): Temporary.
  FL_State newton_state_pert;                   // Q + eps*v, needs halo/ghost slots (like flow_1)
  FL_State newton_residual_pert;                 // R(Q + eps*v), owned cells only (like residual)
  FL_State newton_residual0;                     // R(Q^k), owned cells only
  FL_State newton_rhs;                           // b = R(Q^k) - (Q^k - Q^n)/dt
  FL_State newton_dQ;                            // Newton update
  FL_State krylov_basis[NEWTON_GMRES_M + 1];     // Arnoldi basis V_0..V_m, owned cells only
  FL_State krylov_z;                             // preconditioned scratch
  F32      hessenberg[(NEWTON_GMRES_M + 1) * NEWTON_GMRES_M]; // row-major, stride = NEWTON_GMRES_M
  F32      givens_cs[NEWTON_GMRES_M];
  F32      givens_sn[NEWTON_GMRES_M];
  F32      gmres_g[NEWTON_GMRES_M + 1];
  F32      *reduce_scratch;                      // lane_count()-sized, reused across all reductions

  FL_State flow_0;        // ADD: Q^{n-1}, needed for BDF2
  B32      has_prev_step; // ADD: false until the first backward-Euler bootstrap step completes

} FL_Solver_Euler;

typedef U32 Time_Step_Mode;
enum {
  Time_Step_Global,
  Time_Step_Local,
};
