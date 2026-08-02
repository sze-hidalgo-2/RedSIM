typedef struct FL_Solver_Euler {
  UG_Mesh            *mesh;
  FL_Boundary_Map    *boundary;

  FL_State            flow_1;
  FL_State            flow_2;

  FL_State            residual;

  // NOTE(cmat): Time step buckets for reduction.
  U64                 time_steps_len;
  F64                *time_steps_dat;

  // NOTE(cmat): Halo synchronization.
  IPC_Request_List    halo_request_list;
  U64                 halo_receive_len;
  F32                *halo_receive_dat;
  U64                 halo_send_len;
  F32                *halo_send_dat;
} FL_Solver_Euler;

