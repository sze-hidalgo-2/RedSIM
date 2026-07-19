struct {
  U32 rank_count;
  U32 rank_index;
} IPC_State;

function void ipc_init(void) {
  // NOTE(cmat): Only one thread at a time makes MPI calls, thus MPI_THREAD_SERIALIZED.
  int provided_support = 0;
  MPI_Init_thread((int    *)&sys_context()->command_line.argc,
                  (char ***)&sys_context()->command_line.argv,
                  MPI_THREAD_SERIALIZED, &provided_support);
  
  I32 rank_count = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
  IPC_State.rank_count = (U32)rank_count;

  I32 rank_index = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_index);
  IPC_State.rank_index = (U32)rank_index; 
}

function void ipc_shutdown(void) {
  MPI_Finalize();
}

force_inline function U32 ipc_rank_index (void) { return IPC_State.rank_index; }
force_inline function U32 ipc_rank_count (void) { return IPC_State.rank_count; }

function void ipc_rank_barrier(void) {
  profiler_begin_function();
  if (lane_index() == 0) {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  profiler_end_function();
}

function void log_ipc_context(void) {
  Log_Zone_Scope("IPC Context") {
    log_info("Rank Count: %u", ipc_rank_count());
    log_info("Rank Index: %u", ipc_rank_index());
  }
}
