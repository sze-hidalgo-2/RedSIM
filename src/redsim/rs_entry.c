#include "alice/core/core_build.h"
#include "alice/core/core_build.c"

#include "ipc/ipc_build.h"
#include "ipc/ipc_build.c"

#include "ugrid/ug_build.h"
#include "ugrid/ug_build.c"

#include "ugrid_format/ugf_build.h"
#include "ugrid_format/ugf_build.c"

#if 0

#include "fluid/fl_build.h"
#include "fluid/fl_build.c"

#include "fluid_format/flf_build.h"
#include "fluid_format/flf_build.c"
#endif

#include "alice/linux/linux_system.c"

function void redsim_group_entry(void *user_data) {
  profiler_begin_function();
  log_zone_start("Thread Group Entry");

  // NOTE(cmat): Load mesh on rank 0 and partition it (still multithreaded, just single rank).
  // NTOE(cmat): Once we've loaded the mesh on rank 0, distribute to other ranks and partition again
  // - on each rank for each thread group.
  if (ipc_rank_index() == 0) {
    Arena_Temp scratch = { };
    Scratch_Scope(&scratch, 0) {
      // NOTE(cmat): Load mesh.
      UG_Grid grid = { };
      ugf_grid_init_from_su2(&grid, scratch.arena, str08_lit("cube_30M.su2"));

      // NOTE(cmat): Partition mesh by rank count.
      UG_Partition partition = { };
      ug_partition_rcb(&partition, scratch.arena, &grid, ipc_rank_count());

      // NOTE(cmat): Create a sub-grid for each rank.
    }

    // NOTE(cmat): Construct grid for each partition, distribute across ranks.
  }

  ipc_rank_barrier();

  log_zone_end();
  profiler_end_function();
}

link_function void redsim_entry_point(void) {
  profiler_begin_function();

  log_sys_context();      // NOTE(cmat): Log system information.
  log_ipc_context();      // NOTE(cmat): Log IPC context.
  log_sys_numa_layout();  // NOTE(cmat): Log NUMA layout.

  U32           thread_count = sys_context()->cpu_logical_cores;
  Thread_Group  thread_group = { };

  log_info("Launching thread group with %u threads", thread_count);

  thread_group_init     (&thread_group, str08_lit("Sim_Group"), thread_count);
  thread_group_launch   (&thread_group, redsim_group_entry, 0);
  thread_group_wait_all (&thread_group);
  thread_group_destroy  (&thread_group);

  profiler_end_function();
}

link_function void sys_entry_point(void) {
  // NOTE(cmat): Initialize IPC communication first.
  ipc_init();

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

