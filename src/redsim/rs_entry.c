#include "alice/core/core_build.h"
#include "alice/core/core_build.c"

#include "alice/linux/linux_system.c"

#include "ipc/ipc_build.h"
#include "ipc/ipc_build.c"

#include "ugrid/ug_build.h"
#include "ugrid/ug_build.c"

#include "ugrid_format/ugf_build.h"
#include "ugrid_format/ugf_build.c"

#include "fluid/fl_build.h"
#include "fluid/fl_build.c"

#include "fluid_format/flf_build.h"
#include "fluid_format/flf_build.c"

function void redsim_group_entry(void *user_data) {
  profiler_begin_function();
  log_zone_start("Thread Group Entry");

  Arena permanent_arena = { };
  arena_init(&permanent_arena);

  // NOTE(cmat): Load mesh on rank 0 and partition it (still multithreaded, just single rank).
  // NOTE(cmat): Once we've loaded the mesh on rank 0, distribute to other ranks and partition again
  // - on each rank for each thread group.
  UG_Mesh mesh = { };
  if (ipc_rank_index() == 0) {
    Arena partition_arena = { };
    arena_init(&partition_arena);

    UG_Mesh mesh_global = { };

    // NOTE(cmat): Load grid from file.
    Str08 su2_file = str08_from_cstring((char *)sys_context()->command_line.argv[1]);
    ugf_grid_init_from_su2(&mesh_global.grid, &partition_arena, su2_file);

    // NOTE(cmat): Compute mesh based on grid: adjacency + geometry.
    ug_mesh_init_from_grid(&mesh_global, &partition_arena);

    // NOTE(cmat): Partition mesh by rank count.
    UG_Partition partition = { };
    ug_partition_rcb(&partition, &partition_arena, &mesh_global, ipc_rank_count());

    UG_Mesh_Array mesh_array = { };
    ug_mesh_array_init(&mesh_array, &partition_arena, partition.blocks_len);

    // NOTE(cmat): Create sub-mesh for current rank
    // - Allocated on permanent, since we'll be using this one on this rank.
    ug_mesh_array_from_partition(&mesh_array, &mesh_global, &partition, range1_u64(0, 1), &permanent_arena);

    // NOTE(cmat): Create sub-mesh for each other rank.
    // - Allocated on partition storage, since we'll free after distributing.
    ug_mesh_array_from_partition(&mesh_array, &mesh_global, &partition, range1_u64(1, partition.blocks_len), &partition_arena);

    // NOTE(cmat): Compute cells to send between block for rank 0 mesh (permanent storage).
    ug_mesh_array_compute_sends(&mesh_array, &partition, range1_u64(0, 1), &permanent_arena);

    // NOTE(cmat): Compute cells to send between block for the other ranks (partition storage).
    ug_mesh_array_compute_sends(&mesh_array, &partition, range1_u64(1, partition.blocks_len), &partition_arena);

    // NOTE(cmat): Broadcast mesh array to all ranks.
    ug_mesh_ipc_distribute(&mesh_array);

    // NOTE(cmat): Assign our own mesh to rank 0.
    lane_barrier();
    mesh = mesh_array.dat[0];

    lane_barrier();
    arena_destroy(&partition_arena);
  } else {
    ug_mesh_ipc_receive(&permanent_arena, &mesh, 0);
  }

  // NOTE(cmat): Now, each rank has its own mesh.

  // NOTE(cmat): Compute gradients for each cell.
  ug_mesh_compute_cells_gradient(&mesh, &permanent_arena);

  // NOTE(cmat): Reoder cells by groups: Interior or boundary. [ interior cells | boundary cells ]
  // - This allows us to compute interior cells while waiting for halo cells to be distributed,
  // - needed only by boundary cells.
  ug_mesh_reorder_by_groups(&mesh);

  // NOTE(cmat): Reorder every group in the mesh to improve cache locality.
  ug_mesh_optimize_reorder(&mesh, mesh.groups.cells_interior);
  ug_mesh_optimize_reorder(&mesh, mesh.groups.cells_boundary);

  FL_Solver_Euler solver    = {};
  FL_Boundary_Map boundary  = {};

  F32 wind_angle = f32_pi / 4.f;

  FL_Boundary_Farfield farfield = {
    .density  = 1.225f,
    .velocity = v3f_mul(40.f, v3f(f32_cos(wind_angle), f32_sin(wind_angle), 0)),
    .pressure = 101325.f,
  };

  FL_Material material = {};
  fl_material_init(&material, 1.4f, 1.81e-5f, 0.71f);

  FL_Boundary_Farfield farfield_old = farfield;
  FL_Material          material_old = material;

  FL_Scale ref_scale = {};
  fl_scale_init(&ref_scale, &mesh, farfield.density, farfield.pressure, 1.4f);

  // NOTE(cmat): Normalize all simulated quantities.
  log_info("Normalizing Qualtities");

  Log_Zone_Scope("Reference Scale") {
    log_info("length:      %10.2e", ref_scale.length);
    log_info("density:     %10.2e", ref_scale.density);
    log_info("pressure:    %10.2e", ref_scale.density);
    log_info("sound speed: %10.2e", ref_scale.sound_speed);
  }

  fl_scale_normalize_farfield(&ref_scale, &farfield);
  fl_scale_normalize_material(&ref_scale, &material);

  Log_Zone_Scope("Farfield Boundary") {
    log_info("density:     %10.2e -> %10.2e",               farfield_old.density, farfield.density);
    log_info("velocity x:  %10.2e -> %10.2e",               farfield_old.velocity.x, farfield.velocity.x);
    log_info("velocity y:  %10.2e -> %10.2e",               farfield_old.velocity.y, farfield.velocity.y);
    log_info("velocity z:  %10.2e -> %10.2e",               farfield_old.velocity.z, farfield.velocity.z);
    log_info("pressure:    %10.2e -> %10.2e",               farfield_old.pressure, farfield.pressure);
  }

  Log_Zone_Scope("Material") {
    log_info("gamma:                %10.2e -> %10.2e" , material_old.gamma                , material.gamma);
    log_info("gas_constant:         %10.2e -> %10.2e" , material_old.gas_constant         , material.gas_constant);
    log_info("molecular viscosity:  %10.2e -> %10.2e" , material_old.viscosity_mu         , material.viscosity_mu);
    log_info("thermal conductivity: %10.2e -> %10.2e" , material_old.thermal_conductivity , material.thermal_conductivity);
    log_info("prandtl number:       %10.2e -> %10.2e" , material_old.prandtl_number,        material.prandtl_number);
    log_info("smagorinsky cs:       %10.2e -> %10.2e" , material_old.smagorinsky_cs,        material.smagorinsky_cs);
    log_info("prandtl turbulent:    %10.2e -> %10.2e" , material_old.prandtl_turbulent,     material.prandtl_turbulent);
  }

  log_info("time scaling: %10.2e", fl_scale_denormalize_time(&ref_scale, 1.f))

  // NOTE(cmat): Init boundary map.
  log_info("Initializing boundary");
  fl_boundary_map_init(&boundary, &permanent_arena, 6);
  if (lane_index() == 0) {
    *fl_boundary_map_by_index(&boundary, 0) = (FL_Boundary) { .type = FL_Boundary_Type_No_Slip };
    *fl_boundary_map_by_index(&boundary, 1) = (FL_Boundary) { .type = FL_Boundary_Type_No_Slip };
    *fl_boundary_map_by_index(&boundary, 2) = (FL_Boundary) { .type = FL_Boundary_Type_Farfield, .farfield = farfield };
  }

  // NOTE(cmat): Init solver.
  lane_barrier();

  log_info("Initializing solver");
  fl_solver_euler_init(&solver, &boundary, material, &mesh, &permanent_arena);

  // NOTE(cmat): Initial condition.
  lane_barrier();
  log_info("Initializing flow with farfield");
  fl_state_set_inner_from_farfield(&solver.flow_1, &farfield);

  // NOTE(cmat): Iterate and solve.
  lane_barrier();

  // NOTE(cmat): Export results.
  FLF_Ensight_Export export = { };
  flf_ensight_export_init(&export, str08_lit("sod"), &mesh, &permanent_arena);

  // NOTE(cmat): Compute current gradient + residual for variables using the gradient.
  fl_solver_euler_compute_residual(&solver, &solver.flow_1, &solver.residual, 1);
  flf_ensight_export_flow(&export, &ref_scale, 0.0f, &solver.flow_1, &solver.gradient, solver.cell_time_step);

  F32 time = 0;
  for Iter_Index(it, 100) {
  // for Iter_Index(it, 1) {
    F32 time_step = fl_solver_euler_solve(&solver, 0.f);
    time += fl_scale_denormalize_time(&ref_scale, time_step);

    // NOTE(cmat): Compute current gradient + residual for variables using the gradient.
    fl_solver_euler_compute_residual(&solver, &solver.flow_1, &solver.residual, 1);
    flf_ensight_export_flow(&export, &ref_scale, time, &solver.flow_1, &solver.gradient, solver.cell_time_step);
  }

  log_zone_end();
  profiler_end_function();
}

link_function void redsim_entry_point(void) {
  profiler_begin_function();
  log_info("RedSIM 1.0 | Build Hash: %S", Build_Hash_Str08);
  
  // NOTE(cmat): Check command syntax.
  if (sys_context()->command_line.argc != 2) {
    log_info("Command Syntax: ./redsim_cpu [mesh_file]");
  } else {
    log_sys_context();      // NOTE(cmat): Log system information.
    log_ipc_context();      // NOTE(cmat): Log IPC context.
    log_sys_numa_layout();  // NOTE(cmat): Log NUMA layout.

    U32 thread_count = 0;
    if (sys_numa_layout()->nodes_len > 1) {
      // NOTE(cmat): We are launching thread groups / numa node!
      U64 numa_index = ipc_rank_local_node_index();
      thread_count = sys_numa_layout()->nodes_dat[numa_index].cpus_len;
    } else {
      thread_count = sys_context()->cpu_logical_cores;
    }

    Thread_Group thread_group = { };
    log_info("Launching global thread group with %u threads", thread_count);

    thread_group_init(&thread_group, str08_lit("Sim_Group"), thread_count);

    U64 numa_index = 0;
    if (sys_numa_layout()->nodes_len > 1) {
      numa_index = ipc_rank_local_node_index();
    }

    thread_group_launch   (&thread_group, redsim_group_entry, numa_index, 0);
    thread_group_wait_all (&thread_group);
    thread_group_destroy  (&thread_group);
  }

  profiler_end_function();
}

link_function void sys_entry_point(void) {
  // NOTE(cmat): Initialize IPC communication first.
  ipc_init();

  // NOTE(cmat): Check if RedSIM was launched correctly.
  if (sys_numa_layout()->nodes_len > 1 && ipc_rank_local_node_count() != sys_numa_layout()->nodes_len) {
    log_fatal("Rank count per compute node does not match NUMA domain count: numa = %llu | local ranks = %llu",
              sys_numa_layout()->nodes_len, ipc_rank_local_node_count());
  }

  // NOTE(cmat): Bind main thread to appropriate NUMA node and cpu within that node.
  SYS_CPU bind_to_cpu = 0;
  if (sys_numa_layout()->nodes_len > 1) {
    U64 numa_index            = ipc_rank_local_node_index();
    SYS_NUMA_Node *numa_node  = &sys_numa_layout()->nodes_dat[numa_index];
    bind_to_cpu               = numa_node->cpus_dat[0];
  } else {
    bind_to_cpu = 0;
  }

  sys_thread_bind_to_cpu(bind_to_cpu);

  // NOTE(cmat): Initialize profiler, logger.
  if (ipc_rank_index() == 0) {
    // NOTE(cmat): We only profile for rank 0.
    // - Even though we only profile a single rank, we can still see
    // - the bottlenecks at collective communication calls.
    profiler_startup("spall_trace.spall");
    profiler_init_for_thread();

    // NOTE(cmat): We only generate logs for rank 0.
    // NOTE(cmat): We log both to stdout and files.
    logger_push_hook(logger_write_entry_standard_stream, logger_format_entry_detailed);
  }

  redsim_entry_point();

  // NOTE(cmat): Stop profiling.
  if (ipc_rank_index() == 0){ 
    profiler_quit_for_thread();
    profiler_shutdown();
  }

  // NOTE(cmat): Shutdown IPC.
  ipc_shutdown();
}

