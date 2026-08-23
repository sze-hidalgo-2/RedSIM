typedef struct IPC_MPI_Handle {
  MPI_Request request;
} IPC_MPI_Handle;

typedef struct IPC_MPI_Handle_Node {
  // NOTE(cmat): IPC_Handle_Node base content.
  struct IPC_Handle_Node      *next_list;

  // NOTE(cmat): IPC_Handle_Node extended.
  struct IPC_MPI_Handle_Node  *next;
  struct IPC_MPI_Handle_Node  *prev;

  IPC_MPI_Handle              value;
} IPC_MPI_Handle_Node;

typedef struct IPC_MPI_Handle_List {
  IPC_MPI_Handle_Node *first;
  IPC_MPI_Handle_Node *last;
} IPC_MPI_Handle_List;

global struct {
  U32                 rank_count;
  U32                 rank_index;

  U32                 rank_local_node_count;
  U32                 rank_local_node_index;

  U32                 rank_global_node_count;

  Arena               arena;
  IPC_MPI_Handle_List handle_list;
  IPC_MPI_Handle_List handle_free_list;
} IPC_MPI_State;

function IPC_Handle_Node *ipc_handle_from_mpi_handle(IPC_MPI_Handle_Node *mpi_handle) {
  return (IPC_Handle_Node *)(void *)mpi_handle;
}

function IPC_MPI_Handle_Node *mpi_handle_from_ipc_handle(IPC_Handle_Node *ipc_handle) {
  IPC_MPI_Handle_Node *mpi_handle = (IPC_MPI_Handle_Node *)(void *)ipc_handle;
  return mpi_handle;
}

function IPC_MPI_Handle_Node *ipc_mpi_handle_allocate(void) {
  IPC_MPI_Handle_Node *node = 0;
  if (IPC_MPI_State.handle_free_list.first) {
    node = IPC_MPI_State.handle_free_list.first;
    DLL_Remove_First(IPC_MPI_State.handle_free_list.first, IPC_MPI_State.handle_free_list.last);
    Zero_Fill(node);
    DLL_Push_Back(IPC_MPI_State.handle_list.first, IPC_MPI_State.handle_list.last, node);
  } else {
    node = arena_push_type(&IPC_MPI_State.arena, IPC_MPI_Handle_Node);
    DLL_Push_Back(IPC_MPI_State.handle_list.first, IPC_MPI_State.handle_list.last, node);
  }

  Zero_Fill(&node->value);
  return node;
}

function void ipc_mpi_handle_free(IPC_MPI_Handle_Node *node) {
  DLL_Remove(IPC_MPI_State.handle_list.first, IPC_MPI_State.handle_list.last, node);
  Zero_Fill(node);
  DLL_Push_Back(IPC_MPI_State.handle_free_list.first,  IPC_MPI_State.handle_free_list.last,  node);
}

function void ipc_init(void) {
  if (lane_index() == 0) {
    Zero_Fill(&IPC_MPI_State);
    arena_init(&IPC_MPI_State.arena);

    // NOTE(cmat): Only one thread at a time makes MPI calls, thus MPI_THREAD_SERIALIZED.
    int provided_support = 0;
    int mpi_error = MPI_Init_thread((int    *)&sys_context()->command_line.argc,
                                    (char ***)&sys_context()->command_line.argv,
                                     MPI_THREAD_SERIALIZED, &provided_support);

    if (mpi_error) {
      sys_panic(str08_lit("MPI_Ini_thread error"));
    }
    
    if (provided_support < 2) {
      sys_panic(str08_lit("MPI_Init_thread does not support MPI_THREAD_SERIALIZED"));
    }
    
    I32 rank_count = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    IPC_MPI_State.rank_count = (U32)rank_count;

    I32 rank_index = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_index);
    IPC_MPI_State.rank_index = (U32)rank_index;

    // NOTE(cmat): Build shared communicator.
    MPI_Comm local_node_communicator = 0;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_node_communicator);

    I32 rank_local_node_count = -1;
    MPI_Comm_size(local_node_communicator, &rank_local_node_count);
    IPC_MPI_State.rank_local_node_count = (U32)rank_local_node_count;

    I32 rank_local_node_index = -1;
    MPI_Comm_rank(local_node_communicator, &rank_local_node_index);
    IPC_MPI_State.rank_local_node_index = (U32)rank_local_node_index;

    // NOTE(cmat): Compute total node count.
    // TODO(cmat): This assumes a heterogeneous architecture between nodes, for now.
    IPC_MPI_State.rank_global_node_count = rank_count / rank_local_node_count;

    // NOTE(cmat): Force an early collective communication, in case some implementations do some kind of lazy setup.
    MPI_Barrier(MPI_COMM_WORLD);
  }

  lane_barrier();
}

function void ipc_shutdown(void) {
  if (lane_index() == 0) {
    MPI_Finalize();
  }
}

force_inline function U32 ipc_rank_index (void)             { return IPC_MPI_State.rank_index;              }
force_inline function U32 ipc_rank_count (void)             { return IPC_MPI_State.rank_count;              }
force_inline function U32 ipc_rank_local_node_index (void)  { return IPC_MPI_State.rank_local_node_index;   }
force_inline function U32 ipc_rank_local_node_count (void)  { return IPC_MPI_State.rank_local_node_count;   }
force_inline function U32 ipc_rank_global_node_count (void) { return IPC_MPI_State.rank_global_node_count;  }

function void ipc_rank_barrier(void) {
  profiler_begin_function();
  if (lane_index() == 0) {

    if (ipc_rank_count() > 1) {
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void ipc_rank_request_list_init(IPC_Request_List *request_list) {
  if (lane_index() == 0) {
    Zero_Fill(request_list);
  }

  lane_barrier();
}

function void ipc_rank_request_list_destroy(IPC_Request_List *request_list) {
  if (lane_index() == 0) {
    for (IPC_Request_Node *it = request_list->first; it;) {
      IPC_Request_Node *next = it->next;
      IPC_MPI_Handle_Node *mpi_handle = mpi_handle_from_ipc_handle(it);
      MPI_Request_free(&mpi_handle->value.request);
      ipc_mpi_handle_free(mpi_handle);
      it = next;
    }
  }

  lane_barrier();
}

function void ipc_rank_request_list_start(IPC_Request_List *request_list) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);

  if (lane_index() == 0) {
    for (IPC_Request_Node *it = request_list->first; it; it = it->next) {
      MPI_Start(&mpi_handle_from_ipc_handle(it)->value.request);
    }
  }

  lane_barrier();
  scratch_end(&scratch);
  profiler_end_function();
}

function void ipc_rank_request_list_wait(IPC_Request_List *request_list) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);

  if (lane_index() == 0 && request_list->count) {
    MPI_Request *request_array  = arena_push_count(scratch.arena, MPI_Request, request_list->count);
    U32 request_at              = 0;

    for (IPC_Request_Node *it = request_list->first; it; it = it->next) {
      request_array[request_at++] = mpi_handle_from_ipc_handle(it)->value.request;
    }

    MPI_Waitall((I32)request_list->count, request_array, MPI_STATUSES_IGNORE);
  }

  lane_barrier();
  scratch_end(&scratch);
  profiler_end_function();
}

function void ipc_rank_record_send(IPC_Request_List *request_list, U64 bytes_len, void *bytes_dat, U32 rank, U32 tag) {
  profiler_begin_function();

  if (lane_index() == 0) {
    U64 chunk_count     = bytes_len / (U64)i32_limit_max;
    U64 chunk_remainder = bytes_len % (U64)i32_limit_max;

    U08 *bytes_at = (U08 *)bytes_dat;
    for Iter_Index(it, chunk_count) {
      IPC_MPI_Handle_Node *mpi_handle = ipc_mpi_handle_allocate();
      MPI_Send_init(bytes_at, i32_limit_max, MPI_BYTE, (I32)rank, (I32)tag, MPI_COMM_WORLD, &mpi_handle->value.request);

      IPC_Handle_Node *handle = ipc_handle_from_mpi_handle(mpi_handle);
      Queue_Push(request_list->first, request_list->last, handle);
      request_list->count += 1;

      bytes_at += i32_limit_max;
    }
    
    if (chunk_remainder) {
      IPC_MPI_Handle_Node *mpi_handle = ipc_mpi_handle_allocate();
      MPI_Send_init(bytes_at, (I32)chunk_remainder, MPI_BYTE, (I32)rank, (I32)tag, MPI_COMM_WORLD, &mpi_handle->value.request);

      IPC_Handle_Node *handle = ipc_handle_from_mpi_handle(mpi_handle);
      Queue_Push(request_list->first, request_list->last, handle);
      request_list->count += 1;
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void ipc_rank_record_receive(IPC_Request_List *request_list, U64 bytes_len, void *bytes_dat, U32 rank, U32 tag) {
  profiler_begin_function();

  if (lane_index() == 0) {
    U64 chunk_count     = bytes_len / (U64)i32_limit_max;
    U64 chunk_remainder = bytes_len % (U64)i32_limit_max;

    U08 *bytes_at = (U08 *)bytes_dat;
    for Iter_Index(it, chunk_count) {
      IPC_MPI_Handle_Node *mpi_handle = ipc_mpi_handle_allocate();
      MPI_Recv_init(bytes_at, i32_limit_max, MPI_BYTE, (I32)rank, (I32)tag, MPI_COMM_WORLD, &mpi_handle->value.request);

      IPC_Handle_Node *handle = ipc_handle_from_mpi_handle(mpi_handle);
      Queue_Push(request_list->first, request_list->last, handle);
      request_list->count += 1;

      bytes_at += i32_limit_max;
    }
    
    if (chunk_remainder) {
      IPC_MPI_Handle_Node *mpi_handle = ipc_mpi_handle_allocate();
      MPI_Recv_init(bytes_at, (I32)chunk_remainder, MPI_BYTE, (I32)rank, (I32)tag, MPI_COMM_WORLD, &mpi_handle->value.request);

      IPC_Handle_Node *handle = ipc_handle_from_mpi_handle(mpi_handle);
      Queue_Push(request_list->first, request_list->last, handle);
      request_list->count += 1;
    }
  }

  lane_barrier();
  profiler_end_function();
}

function F64 ipc_rank_minimum_f64(F64 value) {
  profiler_begin_function();

  F64 result = 0;
  if (lane_index() == 0) {
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  }

  lane_broadcast_u64((U64 *)&result, 0);
  profiler_end_function();
  return result;
}

function F64 ipc_rank_sum_f64(F64 value) {
  profiler_begin_function();

  F64 result = 0;
  if (lane_index() == 0) {
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  lane_broadcast_u64((U64 *)&result, 0);
  profiler_end_function();
  return result;
}

function F32 ipc_rank_minimum_f32(F32 value) {
  profiler_begin_function();

  F32 result = 0;
  if (lane_index() == 0) {
    MPI_Allreduce(&value, &result, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  }

  lane_broadcast_type(&result, 0);
  profiler_end_function();
  return result;
}

function void ipc_rank_gather_all_u64(U64 value, U64 *values_all) {
  profiler_begin_function();

  if (lane_index() == 0) {
    MPI_Allgather(&value, sizeof(U64), MPI_BYTE, values_all, sizeof(U64), MPI_BYTE, MPI_COMM_WORLD);
  }

  lane_barrier();
  profiler_end_function();
}

function void log_ipc_context(void) {
  Log_Zone_Scope("IPC Context") {
    log_info("Rank Count:            %u", ipc_rank_count());
    log_info("Rank Index:            %u", ipc_rank_index());
    log_info("Local Node Rank Count: %u", ipc_rank_local_node_count());
    log_info("Local Node Rank Index: %u", ipc_rank_local_node_index());
    log_info("Global Node Count:     %u", ipc_rank_global_node_count());
  }
}

