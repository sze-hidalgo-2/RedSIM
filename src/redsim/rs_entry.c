#include "alice/core/core_build.h"
#include "alice/core/core_build.c"

#include "alice/linux/linux_system.c"

#include "ugrid/ug_build.h"
#include "ugrid/ug_build.c"

#include "ugrid_format/ugf_build.h"
#include "ugrid_format/ugf_build.c"

#include "fluid/fl_build.h"
#include "fluid/fl_build.c"

#include "fluid_format/flf_build.h"
#include "fluid_format/flf_build.c"


function void group_entry(void *user_data) {
  profiler_begin_function();

  local_storage UG_Mesh mesh = { };
  if (lane_index() == 0) {
    ug_mesh_init(&mesh);
    ugf_load_su2(&mesh, str08_lit("cube_4M.su2"));
  }

  lane_barrier();
  ug_mesh_compute_cells       (&mesh);
  ug_mesh_reorder_cells       (&mesh);
  ug_mesh_compute_cells_faces (&mesh);
  ug_mesh_assign_ghosts       (&mesh);

  lane_barrier();

  local_storage FL_Solver_Euler solver = { };
  local_storage FL_Boundary_Map boundary = { };
  // local_storage FL_Boundary_Farfield farfield = { };
  if (lane_index() == 0) {

#if 0
    fl_boundary_map_init(&boundary, 3);

    farfield                                = (FL_Boundary_Farfield)  { .velocity = v3f(50, 0, 0), .density = 1.225f, .pressure = 1.0e5f };
    *fl_boundary_map_by_index(&boundary, 0) = (FL_Boundary)           { .type = FL_Boundary_Type_Farfield, .farfield = farfield };
    *fl_boundary_map_by_index(&boundary, 1) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 2) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
#else
    fl_boundary_map_init(&boundary, 6);
    *fl_boundary_map_by_index(&boundary, 0) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 1) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 2) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 3) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 4) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 5) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
#endif

    fl_solver_euler_init(&solver, &boundary, &mesh);
  }

  // NOTE(cmat): Assign initial condition.
  lane_barrier();

  fl_setup_sod(&solver.flow, solver.mesh);
  // fl_state_set_inner_from_farfield(&solver.flow, &farfield);

  lane_barrier();
  fl_solver_euler_solve(&solver);

  lane_barrier();

  FLF_Ensight_Export export = flf_ensight_export_start(str08_lit("sod_ensight"), &mesh);
  flf_ensight_export_flow(&export, 0.f, &solver.flow);
  flf_ensight_export_end(&export);

  profiler_end_function();
}

link_function void sys_entry_point(void) {
  profiler_begin_function();

  logger_push_hook(logger_write_entry_standard_stream, logger_format_entry_detailed);
  log_sys_context();
  log_sys_numa_layout();

  U32           thread_count = sys_context()->cpu_logical_cores;
  Thread_Group  thread_group = { };

  thread_group_init     (&thread_group, str08_lit("Sim"), thread_count);
  thread_group_launch   (&thread_group, group_entry, 0);
  thread_group_wait_all (&thread_group);
  thread_group_destroy  (&thread_group);

  profiler_end_function();
}

