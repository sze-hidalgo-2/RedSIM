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
} FL_Solver_Euler;

typedef U32 Time_Step_Mode;
enum {
  Time_Step_Global,
  Time_Step_Local,
};
