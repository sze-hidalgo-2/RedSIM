typedef struct FL_Solver_Euler {
  UG_Mesh            *mesh;
  FL_Boundary_Map    *boundary;

  FL_State            flow_1;
  FL_State            flow_2;

  FL_State            residual;

  // NOTE(cmat): Time-Step for each cell.
  F64                *cell_time_step;
  F64                *lane_time_step;

  // NOTE(cmat): Halo synchronization.
  IPC_Request_List    halo_request_list;
  U64                 halo_receive_len;
  F32                *halo_receive_dat;
  U64                 halo_send_len;
  F32                *halo_send_dat;
} FL_Solver_Euler;

typedef U32 Time_Step_Mode;
enum {
  Time_Step_Global,
  Time_Step_Local,
};
