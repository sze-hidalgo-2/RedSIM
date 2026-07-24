typedef struct FL_Solver_Euler {
  UG_Mesh            *mesh;
  FL_Boundary_Map    *boundary;
  FL_State            flow;
  FL_State            residual;

  // NOTE(cmat): Time step buckets for reduction.
  U64                 time_step_bucket_len;
  F64                *time_step_bucket_dat;

  // NOTE(cmat): Halo synchronization.
  IPC_Request_List    halo_request_list;
  U64                 halo_send_len;
  F32                *halo_send_dat;
} FL_Solver_Euler;

function void fl_solver_euler_init          (FL_Solver_Euler *euler, FL_Boundary_Map *boundary, UG_Mesh *mesh, Arena *arena);
function void fl_solver_euler_step_explicit (FL_Solver_Euler *euler, F32 time_step);
function F64  fl_solver_euler_solve_step    (FL_Solver_Euler *euler, F32 CFL);
