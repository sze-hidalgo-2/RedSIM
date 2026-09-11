#define NEWTON_GMRES_M       30     // Krylov restart length
#define NEWTON_GMRES_TOL     1e-2f  // relative — Newton only needs an inexact linear solve
#define NEWTON_MAX_ITERS     10
#define NEWTON_TOL           1e-6f  // relative drop in ||N(Q)|| to accept a Newton step

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
} FL_Solver_Euler;

typedef U32 Time_Step_Mode;
enum {
  Time_Step_Global,
  Time_Step_Local,
};
