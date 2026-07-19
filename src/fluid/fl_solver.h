typedef struct FL_Solver_Euler {
  Arena               arena;
  UG_Mesh            *mesh;
  FL_Boundary_Map    *boundary;
  FL_State            flow;
  FL_State            residual;

  U32                 time_step_bucket_len;
  F64                *time_step_bucket_dat;
} FL_Solver_Euler;

function void fl_solver_euler_init          (FL_Solver_Euler *euler, FL_Boundary_Map *boundary, UG_Mesh *mesh);
function void fl_solver_compute_ghost       (FL_Solver_Euler *euler);
function void fl_solver_euler_step_explicit (FL_Solver_Euler *euler, F32 time_step);
function F64  fl_solver_euler_solve_step    (FL_Solver_Euler *euler, F32 CFL);
