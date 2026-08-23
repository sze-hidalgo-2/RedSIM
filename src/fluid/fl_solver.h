typedef struct FL_Solver_Euler {
  UG_Mesh            *mesh;
  FL_Boundary_Map    *boundary;

  FL_State            flow_1;
  FL_State            flow_2;

  FL_Gradient_State   gradient;
  FL_Limiter_State    limiter;
  FL_State            residual;

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

  // NOTE(cmat): Halo gradient synchronization.
  IPC_Request_List    halo_gradient_request_list;
  U64                 halo_gradient_receive_len;
  F32                *halo_gradient_receive_dat;
  U64                 halo_gradient_send_len;
  F32                *halo_gradient_send_dat;
} FL_Solver_Euler;

typedef U32 Time_Step_Mode;
enum {
  Time_Step_Global,
  Time_Step_Local,
};
