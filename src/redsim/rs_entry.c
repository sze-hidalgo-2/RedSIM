#include "alice/core/core_build.h"
#include "alice/core/core_build.c"

#include "ipc/ipc_build.h"
#include "ipc/ipc_build.c"

#include "ugrid/ug_build.h"
#include "ugrid/ug_build.c"

#include "ugrid_format/ugf_build.h"
#include "ugrid_format/ugf_build.c"

#include "fluid/fl_build.h"
#include "fluid/fl_build.c"

#if 0

#include "fluid_format/flf_build.h"
#include "fluid_format/flf_build.c"

#endif

#include "alice/linux/linux_system.c"

function void redsim_group_entry(void *user_data) {
  profiler_begin_function();
  log_zone_start("Thread Group Entry");

  Arena permanent_arena = { };
  Arena sync_arena      = { };

  arena_init(&permanent_arena);
  arena_init(&sync_arena);

  // NOTE(cmat): Load mesh on rank 0 and partition it (still multithreaded, just single rank).
  // NOTE(cmat): Once we've loaded the mesh on rank 0, distribute to other ranks and partition again
  // - on each rank for each thread group.
  UG_Mesh mesh = { };
  if (ipc_rank_index() == 0) {
    Arena_Temp scratch = scratch_start(0);

    // NOTE(cmat): Load grid from file.
    UG_Grid grid = { };

    Str08 su2_file = str08_from_cstring((char *)sys_context()->command_line.argv[1]);
    ugf_grid_init_from_su2(&grid, scratch.arena, su2_file);

    // NOTE(cmat): Compute mesh based on grid: adjacency + geometry.
    UG_Mesh mesh_global = { };
    ug_mesh_init_from_grid(&mesh_global, &grid, scratch.arena);

    // NOTE(cmat): Partition mesh by rank count.
    UG_Partition partition = { };
    ug_partition_rcb(&partition, scratch.arena, &mesh_global, ipc_rank_count());

    UG_Mesh_Array mesh_array = { };
    ug_mesh_array_init(&mesh_array, scratch.arena, partition.blocks_len);

    // NOTE(cmat): Create sub-mesh for current rank
    // - Allocated on permanent, since we'll be using this one on this rank.
    ug_mesh_array_from_partition(&mesh_array, &mesh_global, &partition, range1_u64(0, 1), &permanent_arena);

    // NOTE(cmat): Create sub-mesh for each other rank.
    // - Allocated on scratch, since we'll free after distributing.
    ug_mesh_array_from_partition(&mesh_array, &mesh_global, &partition, range1_u64(1, partition.blocks_len), scratch.arena);

    // NOTE(cmat): Compute cells to send between block for rank 0 mesh (permanent storage).
    ug_mesh_array_compute_sends(&mesh_array, &partition, range1_u64(0, 1), &permanent_arena);

    // NOTE(cmat): Compute cells to send between block for the other ranks (scratch storage).
    ug_mesh_array_compute_sends(&mesh_array, &partition, range1_u64(1, partition.blocks_len), scratch.arena);

    // NOTE(cmat): Reoder cells by groups: Interior or boundary. [ interior cells | boundary cells ]
    // - This allows us to compute interior cells while waiting for halo cells to be distributed,
    // - needed only by boundary cells.
    for Iter_Index(it, mesh_array.len) {
      ug_mesh_reorder_by_groups(&mesh_array.dat[it]);
    }

    // NOTE(cmat): Reorder every group in the mesh to improve cache locality.
    for Iter_Index(it, mesh_array.len) {
      ug_mesh_optimize_reorder(&mesh_array.dat[it], mesh_array.dat[it].groups.cells_interior);
      ug_mesh_optimize_reorder(&mesh_array.dat[it], mesh_array.dat[it].groups.cells_boundary);
    }

    // NOTE(cmat): Broadcast mesh array to all ranks.
    ug_mesh_ipc_distribute(&mesh_array);

    // NOTE(cmat): Assign our own mesh to rank 0.
    lane_barrier();
    mesh = mesh_array.dat[0];

    lane_barrier();
    scratch_end(&scratch);
  } else {
    ug_mesh_ipc_receive(&permanent_arena, &mesh, 0);
  }

  FL_Solver_Euler solver    = {};
  FL_Boundary_Map boundary  = {};
  
  // NOTE(cmat): Init boundary map.

  log_info("Initializing boundary");
  fl_boundary_map_init(&boundary, &permanent_arena, 6);
  if (lane_index() == 0) {
    *fl_boundary_map_by_index(&boundary, 0) = (FL_Boundary) { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 1) = (FL_Boundary) { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 2) = (FL_Boundary) { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 3) = (FL_Boundary) { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 4) = (FL_Boundary) { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 5) = (FL_Boundary) { .type = FL_Boundary_Type_Slip };
  }

  // NOTE(cmat): Init solver.
  lane_barrier();

  log_info("Initializing solver");
  fl_solver_euler_init(&solver, &boundary, &mesh, &permanent_arena);

  // NOTE(cmat): Initial condition.
  lane_barrier();
  log_info("Initializing flow to SOD condition");
  fl_setup_sod(&solver.flow, &mesh);

  // NOTE(cmat): Iterate and solve.
  lane_barrier();

  // NOTE(cmat): Synchronize all MPI ranks for more accurate
  // - benchmarking measurements for load balacing.
  fl_solver_euler_solve(&solver);

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

    thread_group_init     (&thread_group, str08_lit("Sim_Group"), thread_count);

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
    sys_panic(str08_lit("Rank count per compute node does not match NUMA domain count."));
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

